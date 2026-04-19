"""Post-run analyser for ``chess-commentary-benchmark.json``.

Reads the raw JSON the ``bench_chess_commentary`` script dumps and
produces a quality-vs-speed focused summary. Kept separate from the
benchmark runner so we can re-rank an old run without re-inferring
every reply — useful when the speed-tier thresholds change or when
we want to cross-compare different benchmark invocations side-by-side.

Usage::

    python scripts/analyze_bench_results.py
    python scripts/analyze_bench_results.py path/to/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("docs/documentation/reports/data/chess-commentary-benchmark.json")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"No benchmark JSON at {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())

    models = data.get("models", [])
    rows = []
    for m in models:
        if m.get("error"):
            rows.append((m["slug"], None, None, m["load_time_s"], m["error"]))
            continue
        total_time = 0.0
        total_runs = 0
        for scn in m.get("scenarios", {}).values():
            for t in scn.get("times_s", []):
                total_time += t
            total_runs += len(scn.get("scores", []))
        per_reply = total_time / max(1, total_runs)
        rows.append((m["slug"], m["overall_avg"], per_reply, m["load_time_s"], None))

    # Scoreboard sorted by quality.
    print("# Chess commentary benchmark — quality vs speed")
    print()
    print(f"Source: `{path}`")
    print(f"Config: {data.get('config', {})}")
    print()
    print("## Scoreboard (quality desc)")
    print()
    print("| Model | Quality | Per-reply | Load | Smooth? |")
    print("|---|---|---|---|---|")
    for slug, avg, per, load, err in sorted(rows, key=lambda r: -(r[1] or -1)):
        if err:
            print(f"| `{slug}` | — | — | {load:.1f} | {err} |")
            continue
        ok = avg >= 95.0
        if per < 2.0 and ok:
            tag = "✅ live"
        elif per < 5.0 and ok:
            tag = "🟡 usable"
        elif ok:
            tag = "🔴 slow"
        else:
            tag = "⚠ below quality floor"
        print(f"| `{slug}` | {avg:.1f} | {per:.2f}s | {load:.1f}s | {tag} |")

    print()
    print("## Pareto — best quality per speed tier")
    print()
    print("| Speed tier | Best model | Quality | Per-reply |")
    print("|---|---|---|---|")
    tiers = [
        ("fastest (<1.0 s)", 0.0, 1.0),
        ("live (<2.0 s)", 1.0, 2.0),
        ("usable (<5.0 s)", 2.0, 5.0),
        ("slow (≥5.0 s)", 5.0, 1e9),
    ]
    valid = [(slug, avg, per) for slug, avg, per, _, err in rows if err is None and per is not None]
    for label, lo, hi in tiers:
        in_tier = [(s, a, p) for s, a, p in valid if lo <= p < hi]
        if not in_tier:
            print(f"| {label} | — | — | — |")
            continue
        best = max(in_tier, key=lambda t: t[1])
        print(f"| {label} | `{best[0]}` | {best[1]:.1f} | {best[2]:.2f}s |")

    # Quant-sweep comparison — group by base model via the repo part
    # of `hf:` slugs; show how Q3/Q4/Q5/Q6 rank for the same base.
    print()
    print("## Quant sweep")
    print()
    print("Each family: quality and per-reply by quant bit-depth.")
    print()
    quant_groups: dict[str, list[tuple[str, float, float]]] = {}
    for slug, avg, per, _load, err in rows:
        if err or avg is None:
            continue
        base = _quant_family(slug)
        if base:
            quant_groups.setdefault(base, []).append((slug, avg, per))
    for base, entries in sorted(quant_groups.items()):
        print(f"### {base}")
        print()
        print("| Slug / quant | Quality | Per-reply |")
        print("|---|---|---|")
        for s, a, p in sorted(entries, key=lambda t: _quant_order(t[0])):
            print(f"| `{s}` | {a:.1f} | {p:.2f}s |")
        print()
    return 0


def _quant_family(slug: str) -> str | None:
    """Group quants of the same base model for a side-by-side table.

    Returns a short human label like ``"Gemma 4 E2B"`` or ``"Llama 3.2 1B"``.
    Returns ``None`` for slugs we don't want to cluster (most presets are
    single-quant).
    """
    low = slug.lower()
    if "gemma-4-e2b" in low:
        return "Gemma 4 E2B"
    if "qwen3-1.7b" in low:
        return "Qwen3 1.7B"
    if "llama-3.2-1b" in low:
        return "Llama 3.2 1B"
    if "llama-3.2-3b" in low:
        return "Llama 3.2 3B"
    if "smollm3-3b" in low:
        return "SmolLM3 3B"
    if "qwen2.5-1.5b" in low:
        return "Qwen2.5 1.5B"
    if "qwen2.5-3b" in low:
        return "Qwen2.5 3B"
    return None


def _quant_order(slug: str) -> tuple[int, str]:
    """Sort key that orders by quant bit-depth (ascending). Unrecognised
    quants fall to the end. Returns (bitdepth_rank, fallback_string)."""
    low = slug.lower()
    for i, marker in enumerate(
        [
            "iq2",
            "q2_",
            "iq3",
            "q3_",
            "iq4_xs",
            "q4_0",
            "q4_1",
            "iq4",
            "q4_k_s",
            "q4_k_m",
            "q5_k_s",
            "q5_k_m",
            "q6_k",
            "q8_0",
            "bf16",
        ]
    ):
        if marker in low:
            return (i, slug)
    return (999, slug)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
