"""Multi-model chess-commentary benchmark.

Runs every scenario in :mod:`scripts.eval_llm_commentary` against a
list of candidate LLMs, graded by the same heuristics used during
prompt tuning, and writes a comparison report to
``docs/documentation/reports/chess-commentary-benchmark.md``.

Driven by the same curated scenarios so the headline number you
compare across models is apples-to-apples. Companion to the SLM
tool-calling benchmark (``slm-tool-calling-benchmark.md``) but for
the different axis Rook actually cares about: in-persona commentary
with grounded facts and no hallucinated tactics.

Usage::

    python scripts/bench_chess_commentary.py                  # default model set
    python scripts/bench_chess_commentary.py --models gemma-4-e2b,llama-3.2-3b
    python scripts/bench_chess_commentary.py --temperature 0.3 --repeats 3

Scope choices:

* **One persona** (``casual``) — persona is orthogonal to model choice
  for the metrics we grade on. Adding personas multiplies runtime
  for minimal signal.
* **Generalist + permissive-license models only** — tool-calling
  specialists (xLAM, Hermes) don't help here since
  :class:`MoveInterceptHook` owns chess tools. Non-commercial
  licences are skipped so the default stays safe for the MIT app.
* **Low temperature (0.3)** — leans toward deterministic comparison.
  At higher temperature the same model can swing ±10 points on the
  same scenario, which washes out model-to-model deltas.

Output: ``docs/documentation/reports/chess-commentary-benchmark.md``
with summary table, per-scenario breakdown, and raw reply samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Same-directory import of the eval harness's scenarios + grader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.llm import LLM
from scripts.eval_llm_commentary import (
    Scenario,
    _directive_for,
    _extract_text,
    build_messages,
    grade,
    recompute_with_stockfish,
    scenarios,
)

# Default candidate set — generalists ≤ 4 GB with permissive licenses.
# Skipping tool-calling specialists (xLAM / Hermes / functionary) —
# commentary doesn't use tools, and CC-BY-NC-4.0 models would ship
# restrictions this MIT app doesn't want.
#
# Includes an explicit **quantization sweep** on the current favourite
# (Gemma 4 E2B): Q3_K_M / Q4_K_M / IQ4_XS / Q5_K_M / Q6_K. The BFCL /
# persona-quality delta between Q4 and Q3 is typically ~5-10% on
# instruction-following, paid against a ~300-500 MB download saving —
# worth measuring rather than guessing. ``hf:<repo>:<file>`` shorthand
# bypasses the preset registry so we can pin an exact quant.
_DEFAULT_MODELS = (
    # --- Generalist instruct models @ Q4_K_M ---
    "llama-3.2-1b",
    "llama-3.2-3b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen3-1.7b",
    "gemma-4-e2b",
    "smollm3-3b",
    # --- Tool-calling specialists (permissive licences) @ Q4_K_M ---
    # Pulled from docs/documentation/reports/slm-tool-calling-benchmark.md
    # — strong instruction-followers often help on structured commentary
    # directives even when we don't use their tool-call output.
    "granite-4.0-350m",  # Apache-2.0
    "granite-4.0-1b",  # Apache-2.0
    "hermes-3-3b",  # Llama-3 licence (commercial OK w/ conditions)
    "phi-4-mini",  # MIT
    "hammer-2.1-0.5b",  # Qwen research; check licence per-deploy
    # --- Large specialists (only worth running on a 24 GB card) ---
    "functionary-v3.2",  # MIT, 8 B
    "toolace-2-8b",  # Apache-2.0, 8 B
    # --- Quant sweep on Gemma 4 E2B (current default) ---
    "hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q3_K_M.gguf",
    "hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-IQ4_XS.gguf",
    # gemma-4-e2b = Q4_K_M already listed under generalists
    "hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q5_K_M.gguf",
    "hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q6_K.gguf",
    # --- Quant sweep on Qwen3 1.7B ---
    "hf:unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q3_K_M.gguf",
    "hf:unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q5_K_M.gguf",
    "hf:unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q6_K.gguf",
    # --- Quant sweep on Llama 3.2 1B (Q3 not in the bartowski repo) ---
    "hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q5_K_M.gguf",
    "hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q6_K.gguf",
    # --- Quant sweep on SmolLM3 3B ---
    "hf:unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q3_K_M.gguf",
    "hf:unsloth/SmolLM3-3B-GGUF:SmolLM3-3B-Q5_K_M.gguf",
)


@dataclass
class ModelResult:
    slug: str
    scenario_scores: dict[str, list[int]] = field(default_factory=dict)
    scenario_replies: dict[str, list[str]] = field(default_factory=dict)
    scenario_flags: dict[str, list[list[str]]] = field(default_factory=dict)
    scenario_times: dict[str, list[float]] = field(default_factory=dict)
    load_time_s: float = 0.0
    total_runs: int = 0
    total_score: int = 0
    error: str | None = None

    def record(self, scn_name: str, score: int, reply: str, flags: list[str], elapsed: float) -> None:
        self.scenario_scores.setdefault(scn_name, []).append(score)
        self.scenario_replies.setdefault(scn_name, []).append(reply)
        self.scenario_flags.setdefault(scn_name, []).append(flags)
        self.scenario_times.setdefault(scn_name, []).append(elapsed)
        self.total_runs += 1
        self.total_score += score

    @property
    def overall_avg(self) -> float:
        return self.total_score / max(1, self.total_runs)


def run_model(
    slug: str,
    scns: list[Scenario],
    *,
    persona: str,
    temperature: float,
    max_tokens: int,
    repeats: int,
    n_ctx: int,
) -> ModelResult:
    """Load ``slug``, run every scenario ``repeats`` times, return the
    graded result bundle. All per-scenario replies are kept so the
    report can show qualitative samples next to the score.
    """
    result = ModelResult(slug=slug)
    try:
        t0 = time.perf_counter()
        llm = LLM(model_path=slug, n_ctx=n_ctx)
        result.load_time_s = time.perf_counter() - t0
    except Exception as e:
        result.error = f"load failed: {e!r}"
        return result

    for scn in scns:
        scn.persona = persona
        directive = _directive_for(scn)
        if directive is None:
            # Gate would stay silent — skip this scenario for this
            # model. Record as perfect score so it doesn't drag averages
            # down, with a sentinel flag for transparency.
            result.record(scn.name, 100, "<gated silent>", ["gate silent"], 0.0)
            continue
        for _ in range(repeats):
            messages = build_messages(scn, directive)
            t0 = time.perf_counter()
            try:
                raw = llm.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                elapsed = time.perf_counter() - t0
                reply = _extract_text(raw).strip()
            except Exception as e:
                elapsed = time.perf_counter() - t0
                reply = ""
                result.record(scn.name, 0, f"<error: {e!r}>", [f"complete error: {e!r}"], elapsed)
                continue
            graded = grade(scn, directive, reply)
            result.record(scn.name, graded.score, reply, graded.flags, elapsed)
    return result


def render_report(
    results: list[ModelResult],
    *,
    persona: str,
    temperature: float,
    max_tokens: int,
    n_ctx: int,
    repeats: int,
    scns: list[Scenario],
) -> str:
    """Produce the markdown comparison report. Two sections: the
    headline table (avg score per model), and a per-scenario replies
    block so a reader can judge "clean" vs "catastrophic" failures
    without running the eval themselves.
    """
    lines: list[str] = [
        "# Chess commentary benchmark",
        "",
        "Multi-model comparison of LLM replies against RookApp's commentary "
        "directive. Generated by `scripts/bench_chess_commentary.py` — "
        "re-run it after any gate / prompt change to catch regressions.",
        "",
        f"Run config: persona=`{persona}` · temperature={temperature} · "
        f"max_tokens={max_tokens} · n_ctx={n_ctx} · repeats={repeats}",
        "",
        "## Scoreboard",
        "",
        "| Model | Avg score | Load (s) | Per-reply avg (s) | Smooth? | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.overall_avg if r.error is None else 1):
        if r.error:
            lines.append(f"| `{r.slug}` | — | — | — | — | {r.error} |")
            continue
        per_reply = sum(sum(v) for v in r.scenario_times.values()) / max(1, r.total_runs)
        # Smoothness verdict — Kokoro TTS adds ~1 s before audio lands,
        # so a reply under 2 s keeps the user→Rook loop under 3 s and
        # feels live. 2-5 s is usable; over 5 s the conversational
        # illusion breaks on move-by-move play.
        quality_ok = r.overall_avg >= 95.0
        if per_reply < 2.0 and quality_ok:
            smooth = "✅ live"
        elif per_reply < 5.0 and quality_ok:
            smooth = "🟡 usable"
        elif quality_ok:
            smooth = "🔴 slow"
        else:
            smooth = "⚠ quality floor"
        lines.append(f"| `{r.slug}` | **{r.overall_avg:.1f}** | {r.load_time_s:.1f} | {per_reply:.2f} | {smooth} | — |")
    lines.append("")

    # Pareto table — best model at each latency tier. Users optimising
    # for a smooth game over peak quality pick from here.
    lines.append("### Pareto (quality vs speed)")
    lines.append("")
    lines.append("| Speed tier | Best quality model | Avg score | Per-reply (s) |")
    lines.append("|---|---|---|---|")
    tiers: list[tuple[str, float, float]] = [
        ("fastest (<1.0 s)", 0.0, 1.0),
        ("live (<2.0 s)", 1.0, 2.0),
        ("usable (<5.0 s)", 2.0, 5.0),
        ("slow (≥5.0 s)", 5.0, 1e9),
    ]
    for label, lo, hi in tiers:
        best: ModelResult | None = None
        best_per = 0.0
        for r in results:
            if r.error:
                continue
            per = sum(sum(v) for v in r.scenario_times.values()) / max(1, r.total_runs)
            if lo <= per < hi and (best is None or r.overall_avg > best.overall_avg):
                best = r
                best_per = per
        if best is None:
            lines.append(f"| {label} | — | — | — |")
        else:
            lines.append(f"| {label} | `{best.slug}` | {best.overall_avg:.1f} | {best_per:.2f} |")
    lines.append("")

    lines.append("## Per-scenario scores")
    lines.append("")
    header = ["Model"] + [s.name for s in scns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sorted(results, key=lambda r: -r.overall_avg if r.error is None else 1):
        if r.error:
            continue
        row = [f"`{r.slug}`"]
        for s in scns:
            scores = r.scenario_scores.get(s.name, [])
            if not scores:
                row.append("—")
            else:
                row.append(f"{sum(scores) / len(scores):.0f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Sample replies")
    lines.append("")
    for s in scns:
        lines.append(f"### {s.name}")
        lines.append("")
        lines.append(f"*{s.description}*")
        lines.append("")
        for r in results:
            if r.error:
                continue
            replies = r.scenario_replies.get(s.name, [])
            if not replies:
                continue
            reply = replies[0]  # first rep is enough; extra reps just confirm variance
            flags = r.scenario_flags.get(s.name, [[]])[0]
            flag_str = f"  ⚠ {'; '.join(flags)}" if flags else ""
            lines.append(f"- **`{r.slug}`**: {reply!r}{flag_str}")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Each scenario is a hand-crafted `ChessState` snapshot fed through "
        "`CommentaryGateHook._build_ground_truth` to produce the directive "
        "RookApp would inject at `BEFORE_LLM`. The full system prompt + "
        "directive + user-role task message are fed to the LLM, exactly "
        "as the agent loop would. Replies are graded by the heuristics in "
        "`scripts.eval_llm_commentary.grade`:"
    )
    lines.append("")
    lines.append("- forbidden-term detection (fabricated tactics / pin / fork / …)")
    lines.append("- reply length cap (≤ 40 words — one sentence budget)")
    lines.append("- bare-SAN opener (reply should not start with `Nxd5`)")
    lines.append("- directive-bullet paste (reply verbatim-quotes a fact line)")
    lines.append("- tone mismatch (upbeat-while-losing, rattled-while-winning)")
    lines.append("- `<silent>` sentinel (logged but counted against score)")
    lines.append("- `<think>` tag leakage (would be stripped in pipeline)")
    lines.append("")
    lines.append("Base score 100, minus 12 per flag, clamped to [0, 100].")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append(
        "- [`slm-tool-calling-benchmark.md`](./slm-tool-calling-benchmark) — BFCL-style tool-call benchmark for the same preset pool."
    )
    lines.append(
        "- [`desktop.md`](/documentation/desktop#commentary-quality--evaluation) — design of the commentary gate that produces the directive."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess-commentary benchmark.")
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS))
    parser.add_argument("--persona", default="casual")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument(
        "--output",
        default="docs/documentation/reports/chess-commentary-benchmark.md",
        help="Markdown report path (relative to repo root).",
    )
    parser.add_argument(
        "--json-output",
        default="docs/documentation/reports/data/chess-commentary-benchmark.json",
        help="Raw JSON with every reply + flag + score, for further analysis.",
    )
    parser.add_argument(
        "--no-stockfish",
        action="store_true",
        help="Skip stockfish recomputation of scenario evals (use the hand-set values).",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=12,
        help="Search depth for stockfish eval recomputation.",
    )
    args = parser.parse_args()

    slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    scns = scenarios()
    if not args.no_stockfish:
        # Replay each scenario through stockfish so eval_cp +
        # classification reflect what RookApp would actually see at
        # that position, instead of my eyeballed guesses. Falls open
        # to the hand values when stockfish isn't on $PATH.
        before = sum(1 for s in scns if s.eval_cp is not None)
        scns = recompute_with_stockfish(scns, depth=args.stockfish_depth)
        after = sum(1 for s in scns if s.eval_cp is not None)
        print(f"stockfish recomputed evals: {after}/{len(scns)} scenarios (was {before})")
    results: list[ModelResult] = []
    for slug in slugs:
        print(f"\n{'=' * 78}\n{slug}\n{'=' * 78}", flush=True)
        r = run_model(
            slug,
            scns,
            persona=args.persona,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
            n_ctx=args.n_ctx,
        )
        if r.error:
            print(f"  {r.error}", flush=True)
        else:
            print(f"  overall avg: {r.overall_avg:.1f} / 100 over {r.total_runs} runs", flush=True)
            # Print per-scenario replies inline for real-time review.
            for name, replies in r.scenario_replies.items():
                flags = r.scenario_flags[name][0] if r.scenario_flags[name] else []
                score = r.scenario_scores[name][0] if r.scenario_scores[name] else 0
                flag_str = f"  ⚠ {'; '.join(flags)}" if flags else ""
                print(f"  [{name}] score={score}  reply={replies[0]!r}{flag_str}", flush=True)
        results.append(r)

    # Markdown report.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_report(
            results,
            persona=args.persona,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_ctx=args.n_ctx,
            repeats=args.repeats,
            scns=scns,
        )
    )
    print(f"\nWrote report → {output_path}")

    # JSON dump for downstream analysis.
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "config": {
            "persona": args.persona,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_ctx": args.n_ctx,
            "repeats": args.repeats,
        },
        "models": [
            {
                "slug": r.slug,
                "load_time_s": r.load_time_s,
                "overall_avg": r.overall_avg,
                "total_runs": r.total_runs,
                "error": r.error,
                "scenarios": {
                    name: {
                        "scores": r.scenario_scores.get(name, []),
                        "replies": r.scenario_replies.get(name, []),
                        "flags": r.scenario_flags.get(name, []),
                        "times_s": r.scenario_times.get(name, []),
                    }
                    for name in {n for r in results for n in r.scenario_scores}
                },
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str))
    print(f"Wrote JSON  → {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
