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

    def test_websocket_reset(self, real_server):
        """Verify reset control works with real server."""
        client, _ = real_server
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert "cleared" in msg["message"]
