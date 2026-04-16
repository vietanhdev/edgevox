"""LLM chat via llama-cpp-python, targeting Gemma 4 E2B IT.

Supports both local GGUF files and automatic download from HuggingFace.
Auto-detects GPU layers based on available VRAM.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any

from edgevox.llm.tools import Tool, ToolCallResult, ToolRegistry

log = logging.getLogger(__name__)

# Gemma's chat template sometimes leaks its raw tool-call syntax into
# ``content`` instead of the structured ``tool_calls`` list. We recover
# it here so the agent loop stays reliable regardless of which chat
# format llama-cpp-python picks for the GGUF it happens to load.
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>\s*call:\s*(?P<name>\w+)\s*\{(?P<body>.*?)\}\s*<tool_call\|>",
    re.DOTALL,
)
_GEMMA_QUOTE_RE = re.compile(r"<\|\"\|>")
_KV_PAIR_RE = re.compile(
    r"(?P<k>\w+)\s*[:=]\s*"
    r'(?:"(?P<s>[^"]*)"|'
    r"(?P<n>-?\d+(?:\.\d+)?)|"
    r"(?P<b>true|false|True|False))"
)

# Plain Python-style function call that Gemma sometimes emits instead
# of the templated ``<|tool_call>`` markers. Matches ``name(key="val")``
# or ``name(key=1.2, other="s")``. Deliberately strict (identifier,
# then parenthesised kwargs only) so we don't false-positive on regular
# prose that happens to mention a function.
_PLAIN_CALL_RE = re.compile(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<body>[^()]*)\)")


def _parse_plain_kv_body(body: str) -> dict[str, Any]:
    """Parse ``key="val", other=1, flag=true`` into a dict."""
    args: dict[str, Any] = {}
    for kv in _KV_PAIR_RE.finditer(body):
        key = kv.group("k")
        if kv.group("s") is not None:
            args[key] = kv.group("s")
        elif kv.group("n") is not None:
            num = kv.group("n")
            args[key] = float(num) if "." in num else int(num)
        elif kv.group("b") is not None:
            args[key] = kv.group("b").lower() == "true"
    return args


def _parse_gemma_inline_tool_calls(content: str, known_tools: set[str] | None = None) -> list[dict] | None:
    """Return synthetic ``tool_calls`` entries parsed from raw Gemma
    template markers OR plain ``name(kwargs)`` text, or ``None``.

    Args:
        content: the assistant message body from llama-cpp.
        known_tools: if provided, the plain-call fallback only matches
            names in this set — prevents spurious matches on English
            phrases that happen to look like function calls.
    """
    if not content:
        return None

    calls: list[dict] = []

    # First pass: templated markers
    for idx, match in enumerate(_GEMMA_TOOL_CALL_RE.finditer(content)):
        body = _GEMMA_QUOTE_RE.sub('"', match.group("body"))
        calls.append(
            {
                "id": f"gemma_inline_{idx}",
                "function": {
                    "name": match.group("name"),
                    "arguments": json.dumps(_parse_plain_kv_body(body)),
                },
            }
        )

    if calls:
        return calls

    # Second pass: plain name(args) text — only when we have a tool
    # allowlist to constrain matches.
    if not known_tools:
        return None
    for idx, match in enumerate(_PLAIN_CALL_RE.finditer(content)):
        name = match.group("name")
        if name not in known_tools:
            continue
        body = match.group("body")
        calls.append(
            {
                "id": f"plain_inline_{idx}",
                "function": {
                    "name": name,
                    "arguments": json.dumps(_parse_plain_kv_body(body)),
                },
            }
        )
    return calls or None


DEFAULT_HF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_HF_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"

DEFAULT_PERSONA = "You are Vox, an AI assistant built with EdgeVox."

BASE_VOICE_INSTRUCTIONS = (
    "Keep your responses concise and conversational — aim for 1-3 sentences. "
    "You are talking to the user in real time via voice."
)

# Retained for backwards compatibility with code that imports SYSTEM_PROMPT.
SYSTEM_PROMPT = f"{DEFAULT_PERSONA} {BASE_VOICE_INSTRUCTIONS}"

TOOL_SYSTEM_SUFFIX = (
    " You have tools available. Call a tool only when the user's request needs live data or "
    "an external action; otherwise answer directly. After a tool runs, briefly relay the "
    "result in plain speech — never read JSON aloud."
)

DEFAULT_MAX_TOOL_HOPS = 3

LANGUAGE_HINTS = {
    "vi": "Respond in Vietnamese (tiếng Việt). ",
    "fr": "Respond in French (français). ",
    "es": "Respond in Spanish (español). ",
    "ja": "Respond in Japanese (日本語). ",
    "zh": "Respond in Chinese (中文). ",
    "ko": "Respond in Korean (한국어). ",
    "de": "Respond in German (Deutsch). ",
    "it": "Respond in Italian (italiano). ",
    "pt": "Respond in Portuguese (português). ",
    "hi": "Respond in Hindi (हिन्दी). ",
    "th": "Respond in Thai (ภาษาไทย). ",
    "ru": "Respond in Russian (русский). ",
    "ar": "Respond in Arabic (العربية). ",
    "id": "Respond in Indonesian (Bahasa Indonesia). ",
}


def get_system_prompt(
    language: str = "en",
    has_tools: bool = False,
    persona: str | None = None,
) -> str:
    """Build the LLM system prompt.

    The final string is: ``<language hint> <persona> <voice instructions> <tool suffix?>``.
    Supplying ``persona`` overrides only the identity line so voice-mode
    guidance and language hints are preserved across every agent.
    """
    hint = LANGUAGE_HINTS.get(language, "")
    identity = persona if persona else DEFAULT_PERSONA
    base = f"{hint}{identity} {BASE_VOICE_INSTRUCTIONS}"
    if has_tools:
        base = base + TOOL_SYSTEM_SUFFIX
    return base


def _detect_gpu_layers() -> int:
    """Return number of layers to offload to GPU. 0 = CPU only."""
    from edgevox.core.gpu import get_nvidia_vram_gb, has_metal

    vram_gb = get_nvidia_vram_gb()
    if vram_gb is not None:
        if vram_gb >= 6:
            return -1  # offload all layers
        if vram_gb >= 3:
            return 20
    if has_metal():
        return -1
    return 0


def _resolve_model_path(model_path: str | None) -> str:
    """Return local path to GGUF model, downloading if needed.

    Accepts:
      - Local file path: ``/path/to/model.gguf``
      - HuggingFace shorthand: ``hf:repo/name:filename.gguf``
        e.g. ``hf:unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_K_M.gguf``
      - None → downloads the default Gemma 4 E2B model.
    """
    if model_path and Path(model_path).exists():
        return model_path

    from huggingface_hub import hf_hub_download

    if model_path and model_path.startswith("hf:"):
        parts = model_path[3:].split(":", 1)
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        if not filename:
            raise ValueError(f"HF model path must be 'hf:repo/name:filename', got '{model_path}'")
        log.info(f"Downloading {repo_id}/{filename} ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        log.info(f"Model cached at: {path}")
        return path

    log.info(f"Downloading {DEFAULT_HF_REPO}/{DEFAULT_HF_FILE} ...")
    path = hf_hub_download(
        repo_id=DEFAULT_HF_REPO,
        filename=DEFAULT_HF_FILE,
    )
    log.info(f"Model cached at: {path}")
    return path


ToolCallback = Callable[[ToolCallResult], None]


class LLM:
    """llama-cpp-python chat wrapper with optional tool-calling support."""

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        language: str = "en",
        tools: Iterable[Callable[..., object] | Tool] | ToolRegistry | None = None,
        max_tool_hops: int = DEFAULT_MAX_TOOL_HOPS,
        on_tool_call: ToolCallback | None = None,
        persona: str | None = None,
    ):
        from llama_cpp import Llama

        resolved = _resolve_model_path(model_path)
        n_gpu = _detect_gpu_layers()

        log.info(f"Loading LLM: {resolved} (n_gpu_layers={n_gpu}, n_ctx={n_ctx})")
        self._llm = Llama(
            model_path=resolved,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu,
            verbose=False,
            flash_attn=True,
        )
        self._language = language
        self._registry = self._build_registry(tools)
        self._max_tool_hops = max_tool_hops
        self._on_tool_call = on_tool_call
        # Inference lock — llama_cpp.Llama is NOT thread-safe. Concurrent
        # ``create_chat_completion`` calls from parallel agents must
        # serialize here. Everything else in the agent framework
        # (tool/skill dispatch, bus publishing, workflow composition)
        # runs truly in parallel.
        self._inference_lock = threading.Lock()
        self._history: list[dict] = [
            {"role": "system", "content": get_system_prompt(language, has_tools=bool(self._registry))},
        ]
        log.info(
            "LLM loaded. Tools: %s",
            ", ".join(t.name for t in self._registry) if self._registry else "(none)",
        )

    @staticmethod
    def _build_registry(
        tools: Iterable[Callable[..., object] | Tool] | ToolRegistry | None,
    ) -> ToolRegistry:
        if tools is None:
            return ToolRegistry()
        if isinstance(tools, ToolRegistry):
            return tools
        registry = ToolRegistry()
        registry.register(*tools)
        return registry

    @property
    def tools(self) -> ToolRegistry:
        """The live tool registry. Mutate to add/remove tools at runtime."""
        return self._registry

    def register_tool(self, *funcs: Callable[..., object] | Tool) -> None:
        """Add tools after construction; updates system prompt if needed."""
        was_empty = not self._registry
        self._registry.register(*funcs)
        if was_empty and self._registry:
            self._history[0] = {
                "role": "system",
                "content": get_system_prompt(self._language, has_tools=True),
            }

    def set_language(self, language: str):
        """Update the system prompt for a new language. Keeps conversation history."""
        self._language = language
        self._history[0] = {
            "role": "system",
            "content": get_system_prompt(language, has_tools=bool(self._registry)),
        }

    def _completion_kwargs(self, stream: bool = False) -> dict:
        kwargs: dict = {
            "messages": self._history,
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": stream,
        }
        if self._registry:
            kwargs["tools"] = self._registry.openai_schemas()
            kwargs["tool_choice"] = "auto"
        return kwargs

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        """Thread-safe completion entry point for the agent framework.

        Takes explicit ``messages`` and ``tools`` (no reliance on
        ``self._history`` or ``self._registry``) so concurrent
        ``LLMAgent`` turns stay isolated. Serializes access to the
        underlying ``llama_cpp.Llama`` via :attr:`_inference_lock`
        because llama-cpp is not thread-safe.
        """
        kwargs: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        with self._inference_lock:
            return self._llm.create_chat_completion(**kwargs)

    def _run_agent(self) -> str:
        """Drive the model through tool calls until it produces a text reply.

        Always runs non-streaming. Only used when tools are registered.
        The produced reply is appended to history and returned.
        """
        for hop in range(self._max_tool_hops + 1):
            result = self._llm.create_chat_completion(**self._completion_kwargs(stream=False))
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            raw_content = message.get("content") or ""
            content = raw_content.strip()
            fallback_mode = False

            if not tool_calls:
                fallback_calls = _parse_gemma_inline_tool_calls(raw_content)
                if fallback_calls:
                    log.debug("Recovered %d inline tool call(s) from content", len(fallback_calls))
                    tool_calls = fallback_calls
                    fallback_mode = True
                    # Truncate anything from the first tool marker onward — the model
                    # often hallucinates a tool response + answer after it.
                    cut = raw_content.find("<|tool_call>")
                    content = raw_content[:cut].strip() if cut >= 0 else ""

            if not tool_calls:
                self._history.append({"role": "assistant", "content": content})
                return content

            if hop == self._max_tool_hops:
                log.warning("Tool-call budget exhausted after %d hops", self._max_tool_hops)
                fallback = content or "Sorry, I couldn't finish that request."
                self._history.append({"role": "assistant", "content": fallback})
                return fallback

            results: list[tuple[str, dict]] = []
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                arguments = fn.get("arguments", "{}")
                outcome = self._registry.dispatch(name, arguments)
                if self._on_tool_call is not None:
                    try:
                        self._on_tool_call(outcome)
                    except Exception:
                        log.exception("on_tool_call callback raised")
                payload = (
                    {"ok": True, "result": outcome.result} if outcome.ok else {"ok": False, "error": outcome.error}
                )
                results.append((name, payload))

            if fallback_mode:
                # The chat template didn't emit structured tool_calls, so a
                # ``tool`` role message won't be formatted correctly by
                # llama-cpp's Gemma handler. Feed results back as a user
                # message instead — works with any chat template.
                summary = "; ".join(f"{name} -> {json.dumps(payload, default=str)}" for name, payload in results)
                self._history.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                self._history.append(
                    {
                        "role": "user",
                        "content": f"(system: tool results — {summary}. "
                        f"Now answer the previous request in one short sentence.)",
                    }
                )
            else:
                self._history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                for call, (name, payload) in zip(tool_calls, results, strict=False):
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call_{hop}_{name}"),
                            "name": name,
                            "content": json.dumps(payload, default=str),
                        }
                    )
        return ""  # unreachable

    def chat(self, user_message: str) -> str:
        """Send a message and return the full response."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        if self._registry:
            reply = self._run_agent()
        else:
            result = self._llm.create_chat_completion(**self._completion_kwargs(stream=False))
            reply = (result["choices"][0]["message"].get("content") or "").strip()
            self._history.append({"role": "assistant", "content": reply})
        elapsed = time.perf_counter() - t0

        self._truncate_history()

        log.info(f'LLM: {elapsed:.2f}s → "{reply[:80]}..."')
        return reply

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Stream the final response.

        With no tools registered, streams token-by-token. With tools
        registered, the agent loop runs non-streaming and the final
        reply is emitted as a single chunk — downstream TTS that
        sentence-splits still works naturally.
        """
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        if self._registry:
            reply = self._run_agent()
            if reply:
                yield reply
        else:
            full_reply: list[str] = []
            stream = self._llm.create_chat_completion(**self._completion_kwargs(stream=True))
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content", "")
                if token:
                    full_reply.append(token)
                    yield token
            reply = "".join(full_reply).strip()
            self._history.append({"role": "assistant", "content": reply})

        elapsed = time.perf_counter() - t0
        self._truncate_history()
        log.info(f'LLM stream: {elapsed:.2f}s → "{reply[:80]}..."')

    def _truncate_history(self) -> None:
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

    def reset(self):
        """Clear conversation history."""
        self._history = self._history[:1]
