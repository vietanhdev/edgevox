"""Integration tests for the server module with real models (mocked audio devices).

Tests the FastAPI app, WebSocket protocol, and ServerCore initialization
with actual STT/TTS/LLM models loaded.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def real_server():
    """Create a real ServerCore with smallest models, then build the FastAPI app."""
    try:
        from edgevox.server.core import ServerCore
        from edgevox.server.main import create_app

        core = ServerCore(
            language="en",
            stt_model="tiny",
            stt_device="cpu",
        )
        app = create_app(core)
        client = TestClient(app)
        return client, core
    except Exception as e:
        pytest.skip(f"Server deps not available: {e}")


class TestServerHealthReal:
    def test_health_endpoint(self, real_server):
        client, _ = real_server
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_info_endpoint(self, real_server):
        client, _ = real_server
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "language" in data
        assert "voice" in data
        assert "voices" in data
        assert "tts_sample_rate" in data
        assert data["language"] == "en"


class TestServerWebSocketReal:
    def test_websocket_connect_and_ready(self, real_server):
        client, _ = real_server
        with client.websocket_connect("/ws") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert "session_id" in ready
            assert "languages" in ready
            assert "voices" in ready

    def test_websocket_text_input(self, real_server):
        client, _ = real_server
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            # Send text input
            ws.send_json({"type": "text_input", "text": "Say hello."})

            # Collect messages until we get bot_text
            messages = []
            for _ in range(50):
                try:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg.get("type") == "bot_text":
                        break
                except Exception:
                    break

            types = [m["type"] for m in messages]
            # Should see state changes and bot output
            assert any(t in ("bot_text", "bot_token", "bot_sentence") for t in types)
