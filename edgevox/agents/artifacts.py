"""Shared artifact store for structured agent-to-agent handoffs.

Inspired by Anthropic's harness-design guidance: context resets beat
compaction for long tasks, and *handoffs via files* give sub-agents a
clean context boundary without losing the parent's work.

An :class:`Artifact` is a named, typed blob (text, JSON, or bytes) with
metadata (author agent, created_at, tags). Agents read and write
artifacts via the :class:`ArtifactStore` protocol; a sub-agent spawn
can declare `artifacts=[...]` and the store is rendered into its
system prompt as a lightweight index so it can pull what it needs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)


ArtifactKind = Literal["text", "json", "bytes"]


@dataclass
class Artifact:
    """A named, typed blob with metadata.

    ``version`` is stamped on every write — the in-memory store and
    file-backed store both keep one current artifact per ``name``, and
    the version counter monotonically increments on each overwrite.
    History queries (``store.history(name)``) walk the artifact log.
    Defaults to ``1`` so callers that don't care can ignore the field.
    """

    name: str
    kind: ArtifactKind
    content: Any  # str for text, dict/list for json, bytes for bytes
    author: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    summary: str = ""  # short one-line description for index rendering
    version: int = 1
    parent_version: int | None = None  # set on re-writes; None for v1


@runtime_checkable
class ArtifactStore(Protocol):
    """Thread-safe artifact store."""

    def write(self, artifact: Artifact) -> None: ...
    def read(self, name: str) -> Artifact | None: ...
    def delete(self, name: str) -> bool: ...
    def list(self, *, tag: str | None = None) -> list[Artifact]: ...
    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str: ...


# ---------------------------------------------------------------------------
# InMemoryArtifactStore (default, fast)
# ---------------------------------------------------------------------------


class InMemoryArtifactStore:
    """Dict-backed artifact store. Good for single-process pipelines.

    Auto-versions on overwrite: the current artifact for a name is
    replaced, but its ``version`` counter increments and
    ``parent_version`` points back at the prior. Use :meth:`history`
    to walk the chain of prior versions kept in ``_history``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, Artifact] = {}
        # Append-only history per name. Bounded by ``_max_history`` so a
        # write-heavy pipeline can't slow-leak.
        self._history: dict[str, list[Artifact]] = {}
        self._max_history = 50

    def write(self, artifact: Artifact) -> None:
        with self._lock:
            prior = self._items.get(artifact.name)
            if prior is not None:
                # Auto-version: stamp version = prior.version + 1.
                artifact.parent_version = prior.version
                artifact.version = prior.version + 1
                self._history.setdefault(artifact.name, []).append(prior)
                if len(self._history[artifact.name]) > self._max_history:
                    del self._history[artifact.name][: -self._max_history]
            self._items[artifact.name] = artifact

    def read(self, name: str) -> Artifact | None:
        with self._lock:
            return self._items.get(name)

    def history(self, name: str) -> list[Artifact]:
        """Return prior versions of ``name`` (oldest first). The
        currently active artifact is *not* included — fetch via
        :meth:`read`."""
        with self._lock:
            return list(self._history.get(name, ()))

    def delete(self, name: str) -> bool:
        with self._lock:
            self._history.pop(name, None)
            return self._items.pop(name, None) is not None

    def list(self, *, tag: str | None = None) -> list[Artifact]:
        with self._lock:
            items = list(self._items.values())
            if tag is not None:
                items = [a for a in items if tag in a.tags]
            return items

    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str:
        """Short markdown index — name, one-line summary, tags. Meant
        for injection into a sub-agent's system prompt so it knows what
        it has access to without pre-loading content."""
        items = sorted(self.list(tag=tag), key=lambda a: a.created_at, reverse=True)[:max_items]
        if not items:
            return ""
        lines = ["## Available artifacts"]
        for a in items:
            tagpart = f" [{', '.join(a.tags)}]" if a.tags else ""
            summary = a.summary or _auto_summary(a)
            lines.append(f"- `{a.name}` ({a.kind}){tagpart}: {summary}")
        return "\n".join(lines)


def _auto_summary(a: Artifact) -> str:
    if a.kind == "text" and isinstance(a.content, str):
        first = a.content.strip().splitlines()[0] if a.content.strip() else ""
        return first[:120]
    if a.kind == "json":
        try:
            keys = list(a.content.keys()) if isinstance(a.content, dict) else []
            return f"json with keys: {', '.join(keys[:5])}"
        except Exception:
            return "json blob"
    if a.kind == "bytes":
        size = len(a.content) if isinstance(a.content, (bytes, bytearray)) else 0
        return f"{size} bytes"
    return ""


# ---------------------------------------------------------------------------
# FileArtifactStore (persistent)
# ---------------------------------------------------------------------------


class FileArtifactStore:
    """File-backed store: each artifact is a file under ``base``.

    Text → ``<name>.txt``
    JSON → ``<name>.json``
    Bytes → ``<name>.bin``
    Metadata → ``<name>.meta.json``

    Names may include ``/`` for organization (subdirectories are created).
    """

    def __init__(self, base: str | Path) -> None:
        self.base = Path(base)
        self._lock = threading.RLock()
        # Index cache: name → metadata dict. Avoids the O(files) rglob
        # on every ``list()`` call once the cache is warm. Invalidated
        # on every ``write`` / ``delete`` for that name.
        self._index: dict[str, dict] = {}
        self._index_warm = False

    def _paths(self, name: str, kind: ArtifactKind) -> tuple[Path, Path]:
        ext = {"text": ".txt", "json": ".json", "bytes": ".bin"}[kind]
        main = self.base / f"{name}{ext}"
        meta = self.base / f"{name}.meta.json"
        return main, meta

    def write(self, artifact: Artifact) -> None:
        with self._lock:
            # Auto-version against any existing artifact at this name.
            prior_meta = self._index.get(artifact.name)
            if prior_meta is None and self._index_warm is False:
                # Cold cache: fall back to disk to detect a prior write.
                existing = self.read(artifact.name)
                if existing is not None:
                    prior_meta = {"version": existing.version}
            if prior_meta is not None:
                artifact.parent_version = prior_meta.get("version", 1)
                artifact.version = artifact.parent_version + 1
            main, meta = self._paths(artifact.name, artifact.kind)
            main.parent.mkdir(parents=True, exist_ok=True)
            if artifact.kind == "text":
                main.write_text(artifact.content or "", encoding="utf-8")
            elif artifact.kind == "json":
                main.write_text(json.dumps(artifact.content, indent=2, default=str), encoding="utf-8")
            else:
                main.write_bytes(artifact.content or b"")
            meta_data = {k: v for k, v in asdict(artifact).items() if k != "content"}
            meta.write_text(json.dumps(meta_data, indent=2, default=str), encoding="utf-8")
            # Update index cache.
            self._index[artifact.name] = meta_data

    def read(self, name: str) -> Artifact | None:
        with self._lock:
            for kind in ("text", "json", "bytes"):
                main, meta = self._paths(name, kind)  # type: ignore[arg-type]
                if main.exists() and meta.exists():
                    try:
                        meta_data = json.loads(meta.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if kind == "text":
                        content: Any = main.read_text(encoding="utf-8")
                    elif kind == "json":
                        try:
                            content = json.loads(main.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            content = None
                    else:
                        content = main.read_bytes()
                    return Artifact(content=content, **meta_data)
        return None

    def delete(self, name: str) -> bool:
        with self._lock:
            deleted = False
            for kind in ("text", "json", "bytes"):
                main, meta = self._paths(name, kind)  # type: ignore[arg-type]
                for p in (main, meta):
                    if p.exists():
                        p.unlink()
                        deleted = True
            self._index.pop(name, None)
            return deleted

    def list(self, *, tag: str | None = None) -> list[Artifact]:
        # First call walks the directory to warm the cache; subsequent
        # calls hit the in-memory index. The cache is invalidated on
        # every write/delete for the affected name.
        with self._lock:
            if not self._index_warm:
                self._warm_index()
            names = list(self._index.keys())
        items: list[Artifact] = []
        for name in names:
            a = self.read(name)
            if a is None:
                continue
            if tag is None or tag in a.tags:
                items.append(a)
        return items

    def _warm_index(self) -> None:
        """One-shot walk of the artifact tree to populate the cache."""
        if not self.base.exists():
            self._index_warm = True
            return
        for p in self.base.rglob("*.meta.json"):
            rel = p.relative_to(self.base).with_suffix("").with_suffix("")
            name = str(rel)
            try:
                self._index[name] = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A malformed metadata file shouldn't break the whole
                # listing — skip it.
                continue
        self._index_warm = True

    def history(self, name: str) -> list[Artifact]:
        """File-backed store keeps only the current version (the file
        layout is a single-slot per name). Returns an empty list as a
        Protocol-conforming stub. Use :class:`InMemoryArtifactStore`
        for true history."""
        return []

    def render_index(self, *, tag: str | None = None, max_items: int = 20) -> str:
        items = sorted(self.list(tag=tag), key=lambda a: a.created_at, reverse=True)[:max_items]
        if not items:
            return ""
        lines = ["## Available artifacts"]
        for a in items:
            tagpart = f" [{', '.join(a.tags)}]" if a.tags else ""
            summary = a.summary or _auto_summary(a)
            lines.append(f"- `{a.name}` ({a.kind}){tagpart}: {summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def text_artifact(
    name: str, content: str, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="text", content=content, author=author, tags=list(tags), summary=summary)


def json_artifact(
    name: str, content: Any, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="json", content=content, author=author, tags=list(tags), summary=summary)


def bytes_artifact(
    name: str, content: bytes, *, author: str = "", tags: Iterable[str] = (), summary: str = ""
) -> Artifact:
    return Artifact(name=name, kind="bytes", content=content, author=author, tags=list(tags), summary=summary)


# ---------------------------------------------------------------------------
# Artifact-as-tool exposure (Anthropic memory-tool / Claude Agent SDK pattern)
# ---------------------------------------------------------------------------


def make_artifact_tools(store: ArtifactStore) -> list:
    """Return ``@tool``-decorated functions exposing ``store`` to an LLM.

    Three tools, named to be self-documenting in tool-call traces:

    - ``read_artifact(name)`` → JSON-encoded ``{kind, content, version}``
      or ``{ok: false, error: ...}`` when the artifact is missing.
      Bytes artifacts return their byte length only — the model
      shouldn't pull binary blobs into the prompt.
    - ``write_artifact(name, content, kind="text", summary="")`` →
      writes a new artifact (or auto-versions an overwrite). Returns
      the assigned ``version``.
    - ``list_artifacts(tag="")`` → a short markdown index suitable for
      one-shot prompt rendering.

    Pairs with :meth:`ArtifactStore.render_index` — the index gives
    the model an at-a-glance catalog, the tools let it actually pull
    contents on demand instead of paying the prompt cost every turn.
    Inspired by Anthropic's memory tool / Claude Agent SDK skills:
    "filesystem is the API, the model navigates it."
    """
    from edgevox.llm.tools import tool

    @tool
    def read_artifact(name: str) -> dict:
        """Read an artifact by name. Returns its kind, version, and content
        (or an error if missing). Binary artifacts return only their length.

        Args:
            name: artifact name as registered in the store.
        """
        a = store.read(name)
        if a is None:
            return {"ok": False, "error": f"no artifact named {name!r}"}
        if a.kind == "bytes":
            content_repr: Any = {"bytes_len": len(a.content) if isinstance(a.content, (bytes, bytearray)) else 0}
        else:
            content_repr = a.content
        return {
            "ok": True,
            "name": a.name,
            "kind": a.kind,
            "version": a.version,
            "content": content_repr,
            "tags": list(a.tags),
            "summary": a.summary,
        }

    @tool
    def write_artifact(name: str, content: str, kind: str = "text", summary: str = "") -> dict:
        """Write (or overwrite) an artifact. Auto-versions on overwrite.

        Args:
            name: artifact name. Slashes create nested groups in file stores.
            content: artifact body. ``str`` for text/json, hex string for bytes.
            kind: one of ``text``, ``json``, ``bytes``. Defaults to ``text``.
            summary: short one-line description for the artifact index.
        """
        if kind not in ("text", "json", "bytes"):
            return {"ok": False, "error": f"invalid kind {kind!r}"}
        body: Any = content
        if kind == "json":
            try:
                body = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"invalid json content: {e}"}
        elif kind == "bytes":
            try:
                body = bytes.fromhex(content) if isinstance(content, str) else content
            except ValueError as e:
                return {"ok": False, "error": f"invalid hex content: {e}"}
        a = Artifact(name=name, kind=kind, content=body, summary=summary)
        store.write(a)
        return {"ok": True, "name": name, "version": a.version}

    @tool
    def list_artifacts(tag: str = "") -> str:
        """Return a markdown index of available artifacts, optionally
        filtered by tag.

        Args:
            tag: filter to artifacts carrying this tag. Empty = all.
        """
        return store.render_index(tag=tag or None)

    return [read_artifact, write_artifact, list_artifacts]


__all__ = [
    "Artifact",
    "ArtifactKind",
    "ArtifactStore",
    "FileArtifactStore",
    "InMemoryArtifactStore",
    "bytes_artifact",
    "json_artifact",
    "make_artifact_tools",
    "text_artifact",
]
