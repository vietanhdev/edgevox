"""ROS2 ActionServer for skill invocation.

Exposes a single ``edgevox_msgs/action/ExecuteSkill`` action at
``<namespace>/execute_skill``. External ROS2 action clients send a
goal with ``skill_name`` + ``arguments_json``, receive feedback as JSON,
and get a terminal result with ``ok`` + ``result_json``.

``edgevox_msgs`` is a separately-built colcon interface package. This
module imports it lazily — if the package hasn't been built + sourced
the adapter gracefully no-ops (returns ``None`` from the factory).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from edgevox.agents.skills import GoalStatus

logger = logging.getLogger(__name__)

try:
    import rclpy  # noqa: F401
    from rclpy.action import ActionServer, CancelResponse, GoalResponse

    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False

try:
    from edgevox_msgs.action import ExecuteSkill  # type: ignore[import-not-found]

    _EDGEVOX_MSGS_AVAILABLE = True
except Exception:  # covers ImportError + weird rosidl edge cases
    _EDGEVOX_MSGS_AVAILABLE = False


# Type alias for a skill dispatcher function. Receives
# ``(skill_name, kwargs)`` and returns a ``GoalHandle`` (EdgeVox's
# in-process one, not the ROS2 action goal handle).
SkillDispatcher = Callable[[str, dict[str, Any]], Any]


class SkillActionServer:
    """Exposes EdgeVox skills to external ROS2 action clients.

    Args:
        node: the ``rclpy.Node`` (typically ``ROS2Bridge._node``).
        dispatch: function that takes ``(skill_name, arguments dict)``
            and returns a :class:`~edgevox.agents.skills.GoalHandle`.
            Typically wraps ``agent.run`` or a direct
            ``SimEnvironment.apply_action`` call.
    """

    def __init__(self, node: Any, dispatch: SkillDispatcher) -> None:
        if not _RCLPY_AVAILABLE:
            raise RuntimeError("rclpy not available")
        if not _EDGEVOX_MSGS_AVAILABLE:
            raise RuntimeError(
                "edgevox_msgs not available — build with `colcon build --packages-select edgevox_msgs` "
                "and source install/setup.bash"
            )

        self._node = node
        self._dispatch = dispatch
        self._server = ActionServer(
            node,
            ExecuteSkill,
            "execute_skill",
            execute_callback=self._execute,
            cancel_callback=self._on_cancel,
            goal_callback=self._on_goal,
        )
        logger.info("SkillActionServer up at <ns>/execute_skill")

    # ----- shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self._server.destroy()
        except Exception:
            logger.debug("action server destroy failed", exc_info=True)

    # ----- goal / cancel callbacks ----------------------------------------

    @staticmethod
    def _on_goal(_goal_request: Any) -> Any:
        return GoalResponse.ACCEPT

    @staticmethod
    def _on_cancel(_goal_handle: Any) -> Any:
        return CancelResponse.ACCEPT

    # ----- execution ------------------------------------------------------

    def _execute(self, goal_handle: Any) -> Any:
        goal = goal_handle.request
        skill = goal.skill_name
        raw_args = goal.arguments_json or "{}"
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments_json must decode to an object")
        except Exception as e:
            goal_handle.abort()
            return _make_result(ok=False, result="", error=f"bad arguments_json: {e}")

        try:
            handle = self._dispatch(skill, args)
        except Exception as e:
            logger.exception("skill dispatcher raised")
            goal_handle.abort()
            return _make_result(ok=False, result="", error=str(e))

        if handle is None:
            goal_handle.abort()
            return _make_result(ok=False, result="", error="dispatcher returned None")

        feedback_msg = ExecuteSkill.Feedback()

        # Block in the execute callback until the skill reaches a
        # terminal state, periodically draining feedback and honouring
        # ROS2 cancel requests.
        while True:
            if goal_handle.is_cancel_requested:
                handle.cancel()
            for fb in handle.feedback():
                try:
                    feedback_msg.feedback_json = json.dumps(fb, default=str)
                    goal_handle.publish_feedback(feedback_msg)
                except Exception:
                    logger.debug("publish_feedback failed", exc_info=True)
            status = handle.poll(timeout=0.1)
            if status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
                # Drain any trailing feedback queued just before terminal.
                for fb in handle.feedback():
                    try:
                        feedback_msg.feedback_json = json.dumps(fb, default=str)
                        goal_handle.publish_feedback(feedback_msg)
                    except Exception:
                        logger.debug("publish_feedback failed", exc_info=True)
                break

        if handle.status is GoalStatus.SUCCEEDED:
            goal_handle.succeed()
            return _make_result(
                ok=True,
                result=json.dumps(handle.result, default=str) if handle.result is not None else "",
                error="",
            )
        if handle.status is GoalStatus.CANCELLED:
            goal_handle.canceled()
            return _make_result(ok=False, result="", error="cancelled")
        goal_handle.abort()
        return _make_result(ok=False, result="", error=str(handle.error or "skill failed"))


def _make_result(*, ok: bool, result: str, error: str) -> Any:
    msg = ExecuteSkill.Result()
    msg.ok = ok
    msg.result_json = result
    msg.error = error
    return msg


def create_skill_action_server(node: Any, dispatch: SkillDispatcher) -> SkillActionServer | None:
    """Factory that returns ``None`` when edgevox_msgs isn't built."""
    if not _RCLPY_AVAILABLE or not _EDGEVOX_MSGS_AVAILABLE:
        return None
    try:
        return SkillActionServer(node, dispatch)
    except Exception:
        logger.exception("SkillActionServer init failed")
        return None


def is_available() -> bool:
    """True iff rclpy + edgevox_msgs are both importable."""
    return _RCLPY_AVAILABLE and _EDGEVOX_MSGS_AVAILABLE


__all__ = ["SkillActionServer", "create_skill_action_server", "is_available"]
