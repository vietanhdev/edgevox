"""Tests for InMemoryArtifactStore and FileArtifactStore."""

from __future__ import annotations

import pytest

from edgevox.agents.artifacts import (
    FileArtifactStore,
    InMemoryArtifactStore,
    bytes_artifact,
    json_artifact,
    text_artifact,
)


@pytest.fixture(params=["mem", "file"])
def store(request, tmp_path):
    if request.param == "mem":
        return InMemoryArtifactStore()
    return FileArtifactStore(tmp_path / "art")


def test_text_roundtrip(store):
    a = text_artifact("plan.md", "# plan\nstep 1", author="planner", tags=["plan"])
    store.write(a)
    got = store.read("plan.md")
    assert got is not None
    assert got.content == "# plan\nstep 1"
    assert got.author == "planner"
    assert "plan" in got.tags


def test_json_roundtrip(store):
    a = json_artifact("state.json", {"x": 1, "y": [1, 2, 3]})
    store.write(a)
    got = store.read("state.json")
    assert got is not None
    assert got.content == {"x": 1, "y": [1, 2, 3]}


def test_bytes_roundtrip(store):
    a = bytes_artifact("blob", b"\x00\x01\x02")
    store.write(a)
    got = store.read("blob")
    assert got is not None
    assert got.content == b"\x00\x01\x02"


def test_list_and_delete(store):
    store.write(text_artifact("a", "1"))
    store.write(text_artifact("b", "2"))
    names = {a.name for a in store.list()}
    assert names == {"a", "b"}
    assert store.delete("a") is True
    assert {a.name for a in store.list()} == {"b"}


def test_list_by_tag(store):
    store.write(text_artifact("a", "1", tags=["plan"]))
    store.write(text_artifact("b", "2", tags=["result"]))
    plans = store.list(tag="plan")
    assert [a.name for a in plans] == ["a"]


def test_read_missing(store):
    assert store.read("nope") is None


def test_render_index_format(store):
    store.write(text_artifact("plan.md", "step 1\nstep 2", tags=["plan"], summary="two-step plan"))
    idx = store.render_index()
    assert "plan.md" in idx
    assert "two-step plan" in idx


def test_render_index_empty(store):
    assert store.render_index() == ""


def test_overwrite(store):
    store.write(text_artifact("a", "first"))
    store.write(text_artifact("a", "second"))
    got = store.read("a")
    assert got.content == "second"


def test_file_store_survives_reopen(tmp_path):
    store = FileArtifactStore(tmp_path / "art")
    store.write(text_artifact("persistent", "hello"))
    store2 = FileArtifactStore(tmp_path / "art")
    assert store2.read("persistent").content == "hello"


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_overwrite_auto_versions_in_memory(mem_artifact_store):
    store = mem_artifact_store
    store.write(text_artifact("plan", "v1"))
    store.write(text_artifact("plan", "v2"))
    store.write(text_artifact("plan", "v3"))
    current = store.read("plan")
    assert current.content == "v3"
    assert current.version == 3
    assert current.parent_version == 2
    history = store.history("plan")
    assert [a.content for a in history] == ["v1", "v2"]
    assert [a.version for a in history] == [1, 2]


def test_overwrite_auto_versions_in_file(tmp_path):
    store = FileArtifactStore(tmp_path / "art")
    store.write(text_artifact("plan", "v1"))
    store.write(text_artifact("plan", "v2"))
    current = store.read("plan")
    assert current.content == "v2"
    assert current.version == 2
    assert current.parent_version == 1


# ---------------------------------------------------------------------------
# Index cache (FileArtifactStore)
# ---------------------------------------------------------------------------


def test_file_store_list_uses_index_cache(tmp_path, monkeypatch):
    store = FileArtifactStore(tmp_path / "art")
    store.write(text_artifact("a", "A"))
    store.write(text_artifact("b", "B"))
    # Warm the cache.
    assert {a.name for a in store.list()} == {"a", "b"}

    # Subsequent calls must not rglob the disk again — patch rglob to
    # raise so we can prove the cache short-circuits the walk.
    from pathlib import Path

    def _boom(self, *args, **kw):
        raise AssertionError("rglob should not be called once cache is warm")

    monkeypatch.setattr(Path, "rglob", _boom)
    # This should still succeed via the cache.
    assert {a.name for a in store.list()} == {"a", "b"}


# ---------------------------------------------------------------------------
# Artifact-as-tool exposure
# ---------------------------------------------------------------------------


def test_make_artifact_tools_returns_three_tools(mem_artifact_store):
    from edgevox.agents.artifacts import make_artifact_tools

    tools = make_artifact_tools(mem_artifact_store)
    names = {getattr(t, "__edgevox_tool__", None).name for t in tools}
    assert names == {"read_artifact", "write_artifact", "list_artifacts"}


def test_artifact_tools_round_trip(mem_artifact_store):
    from edgevox.agents.artifacts import make_artifact_tools

    read_tool, write_tool, list_tool = make_artifact_tools(mem_artifact_store)
    # Write a text artifact via the tool.
    write_result = write_tool(name="plan", content="step 1: scan", kind="text", summary="initial plan")
    assert write_result["ok"] is True
    assert write_result["version"] == 1

    # Read it back.
    read_result = read_tool(name="plan")
    assert read_result["ok"] is True
    assert read_result["content"] == "step 1: scan"
    assert read_result["version"] == 1
    assert read_result["summary"] == "initial plan"

    # Listing returns the markdown index.
    idx = list_tool(tag="")
    assert "plan" in idx
    assert "## Available artifacts" in idx


def test_artifact_tools_handle_missing(mem_artifact_store):
    from edgevox.agents.artifacts import make_artifact_tools

    read_tool, _w, _l = make_artifact_tools(mem_artifact_store)
    out = read_tool(name="ghost")
    assert out["ok"] is False
    assert "ghost" in out["error"]


def test_artifact_tools_rejects_bad_kind(mem_artifact_store):
    from edgevox.agents.artifacts import make_artifact_tools

    _r, write_tool, _l = make_artifact_tools(mem_artifact_store)
    out = write_tool(name="x", content="anything", kind="audio")
    assert out["ok"] is False
    assert "invalid kind" in out["error"]


def test_artifact_tools_bytes_returns_length_only(mem_artifact_store):
    """The model must not pull binary blobs into its prompt — ``read``
    on a bytes artifact returns only the length."""
    from edgevox.agents.artifacts import bytes_artifact, make_artifact_tools

    mem_artifact_store.write(bytes_artifact("snap", b"\x00\x01\x02\x03"))
    read_tool, _w, _l = make_artifact_tools(mem_artifact_store)
    out = read_tool(name="snap")
    assert out["ok"] is True
    assert out["kind"] == "bytes"
    assert out["content"] == {"bytes_len": 4}
