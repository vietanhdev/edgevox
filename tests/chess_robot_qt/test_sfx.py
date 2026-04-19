"""SAN → sound classifier unit tests.

Exercises :func:`classify_move_sfx` across the move shapes the window
will see from :class:`ChessState.last_move_san`. The classifier is a
pure function so we don't need PortAudio or Qt here; audio playback
is covered by manual testing on the app itself.
"""

from __future__ import annotations

from edgevox.apps.chess_robot_qt.sfx import classify_move_sfx


def test_plain_move():
    assert classify_move_sfx("e4") == "move"
    assert classify_move_sfx("Nf3") == "move"


def test_capture():
    assert classify_move_sfx("exd5") == "capture"
    assert classify_move_sfx("Qxf7") == "capture"


def test_check_not_mate():
    """``+`` suffix fires the check cue even when the underlying
    action is a capture — the attention signal matters more than the
    sub-type."""
    assert classify_move_sfx("Qxf7+") == "check"
    assert classify_move_sfx("Bb5+") == "check"


def test_checkmate_maps_to_game_end():
    """``#`` always means the game just ended, regardless of
    ``is_game_over`` — the two flags are redundant on purpose (env
    may set ``is_game_over`` a beat after the SAN settles)."""
    assert classify_move_sfx("Qxf7#") == "game_end"


def test_game_over_override():
    """Stalemate / insufficient material: no SAN marker but
    ``is_game_over`` must still fire the end-of-game chime."""
    assert classify_move_sfx("Kh1", is_game_over=True) == "game_end"
    assert classify_move_sfx(None, is_game_over=True) == "game_end"


def test_castle():
    assert classify_move_sfx("O-O") == "castle"
    assert classify_move_sfx("O-O-O") == "castle"


def test_castle_with_check_suffix_routes_to_check():
    """Castling-into-check (rare but legal): the check cue takes
    priority over the castle cue because it's the more actionable
    signal for the listener."""
    assert classify_move_sfx("O-O+") == "check"


def test_empty_san_without_game_over_plays_plain_move():
    """Defensive: state could briefly expose ``last_move_san=None``
    during rehydration. Play the plain-move click rather than crash."""
    assert classify_move_sfx(None) == "move"
    assert classify_move_sfx("") == "move"


def test_promotion_is_plain_move_without_capture():
    """Promotion without capture is still a move. Capture-promotion
    routes through the capture cue because ``x`` is present."""
    assert classify_move_sfx("e8=Q") == "move"
    assert classify_move_sfx("exd8=Q") == "capture"
    assert classify_move_sfx("exd8=Q+") == "check"
