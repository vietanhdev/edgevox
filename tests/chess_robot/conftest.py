"""Shared fixtures for Rook (chess_robot) tests."""

from __future__ import annotations

import pytest

from edgevox.integrations.chess.environment import ChessEnvironment
from tests.chess.conftest import FakeEngine


@pytest.fixture
def env():
    """ChessEnvironment with the deterministic FakeEngine — user plays white."""
    e = ChessEnvironment(FakeEngine(), user_plays="white")
    try:
        yield e
    finally:
        e.close()


@pytest.fixture
def env_engine_white():
    """ChessEnvironment where the engine plays white (inverted POV)."""
    e = ChessEnvironment(FakeEngine(), user_plays="black")
    try:
        yield e
    finally:
        e.close()
