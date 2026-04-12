"""ROS2 bridge for EdgeVox — publishes voice pipeline events to ROS2 topics.

Optional: requires 'rclpy' from a sourced ROS2 workspace (not available on PyPI).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import ROS2 dependencies — they are entirely optional.
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from std_msgs.msg import Float32, String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ---------------------------------------------------------------------------
# QoS profiles tailored to each topic category
# ---------------------------------------------------------------------------
# State: late joiners need the latest value immediately
_QOS_STATE = (
    QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    if ROS2_AVAILABLE
    else None
)

# High-frequency sensor-like data (audio level, streaming tokens): drop stale samples
_QOS_SENSOR = (
    QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )
    if ROS2_AVAILABLE
    else None
)

# Reliable event/command topics: messages must not be lost
_QOS_RELIABLE = (
    QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    if ROS2_AVAILABLE
    else None
)


# ---------------------------------------------------------------------------
# ROS2Bridge — the real implementation (used when rclpy is present)
# ---------------------------------------------------------------------------
class ROS2Bridge:
    """Thin wrapper that exposes the voice pipeline to ROS2 topics.

    All topic names are **relative** — the actual prefix is determined by the
    node namespace (default ``/edgevox``).  For example, with namespace
    ``/robot1/edgevox`` the transcription topic becomes
    ``/robot1/edgevox/transcription``.

    Publishers
    ----------
    transcription   (std_msgs/String)   user speech text
    response        (std_msgs/String)   bot full response text
    state           (std_msgs/String)   pipeline state  [TRANSIENT_LOCAL]
    audio_level     (std_msgs/Float32)  mic audio level 0.0-1.0  [BEST_EFFORT]
    metrics         (std_msgs/String)   JSON-encoded latency metrics
    bot_token       (std_msgs/String)   streaming LLM tokens  [BEST_EFFORT]
    bot_sentence    (std_msgs/String)   completed sentences (TTS chunks)
    wakeword        (std_msgs/String)   wake word detection events
    info            (std_msgs/String)   JSON responses to query commands

    Subscribers
    -----------
    tts_request     (std_msgs/String)   external text-to-speak
    command         (std_msgs/String)   commands: reset / mute / unmute /
                                        list_voices / list_languages /
                                        hardware_info / model_info
    text_input      (std_msgs/String)   text → LLM (bypass STT)
    interrupt       (std_msgs/String)   interrupt current response
    set_language    (std_msgs/String)   switch language (ISO 639-1 code)
    set_voice       (std_msgs/String)   switch TTS voice

    Parameters
    ----------
    language        (string)  current language ISO 639-1 code
    voice           (string)  current TTS voice name
    muted           (bool)    whether the microphone is muted
    """

    def __init__(self, *, namespace: str = "/edgevox") -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy is not installed. Use create_bridge() for graceful fallback.")

        # Initialise rclpy if it hasn't been already.
        if not rclpy.ok():
            rclpy.init()

        self._node: Node = rclpy.create_node("edgevox", namespace=namespace)
        logger.info("ROS2 node 'edgevox' created (namespace=%s)", namespace)

        # -- Declare parameters ------------------------------------------------
        self._node.declare_parameter("language", "en")
        self._node.declare_parameter("voice", "")
        self._node.declare_parameter("muted", False)
        self._node.add_on_set_parameters_callback(self._on_param_change)

        # -- Publishers (relative topic names) ---------------------------------
        self._pub_transcription = self._node.create_publisher(String, "transcription", _QOS_RELIABLE)
        self._pub_response = self._node.create_publisher(String, "response", _QOS_RELIABLE)
        self._pub_state = self._node.create_publisher(String, "state", _QOS_STATE)
        self._pub_audio_level = self._node.create_publisher(Float32, "audio_level", _QOS_SENSOR)
        self._pub_metrics = self._node.create_publisher(String, "metrics", _QOS_RELIABLE)
        self._pub_bot_token = self._node.create_publisher(String, "bot_token", _QOS_SENSOR)
        self._pub_bot_sentence = self._node.create_publisher(String, "bot_sentence", _QOS_RELIABLE)
        self._pub_wakeword = self._node.create_publisher(String, "wakeword", _QOS_RELIABLE)
        self._pub_info = self._node.create_publisher(String, "info", _QOS_RELIABLE)

        # Callbacks for subscribers (set externally via set_*_callback)
        self._tts_callback: Callable[[str], Any] | None = None
        self._command_callback: Callable[[str], Any] | None = None
        self._text_input_callback: Callable[[str], Any] | None = None
        self._interrupt_callback: Callable[[], Any] | None = None
        self._set_language_callback: Callable[[str], Any] | None = None
        self._set_voice_callback: Callable[[str], Any] | None = None
        self._query_callback: Callable[[str], dict | None] | None = None

        # -- Subscribers (relative topic names) --------------------------------
        self._sub_tts_request = self._node.create_subscription(
            String, "tts_request", self._on_tts_request, _QOS_RELIABLE
        )
        self._sub_command = self._node.create_subscription(String, "command", self._on_command, _QOS_RELIABLE)
        self._sub_text_input = self._node.create_subscription(String, "text_input", self._on_text_input, _QOS_RELIABLE)
        self._sub_interrupt = self._node.create_subscription(String, "interrupt", self._on_interrupt, _QOS_RELIABLE)
        self._sub_set_language = self._node.create_subscription(
            String, "set_language", self._on_set_language, _QOS_RELIABLE
        )
        self._sub_set_voice = self._node.create_subscription(String, "set_voice", self._on_set_voice, _QOS_RELIABLE)

        self._spin_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    # -- parameter handling ----------------------------------------------------

    def _on_param_change(self, params: list) -> Any:
        """Handle parameter changes from ``ros2 param set``."""
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == "language" and param.type_ == Parameter.Type.STRING:
                logger.info("Parameter 'language' changed to %s", param.value)
                if self._set_language_callback is not None:
                    try:
                        self._set_language_callback(param.value)
                    except Exception:
                        logger.exception("Error in set_language callback (param)")
            elif param.name == "voice" and param.type_ == Parameter.Type.STRING:
                logger.info("Parameter 'voice' changed to %s", param.value)
                if self._set_voice_callback is not None:
                    try:
                        self._set_voice_callback(param.value)
                    except Exception:
                        logger.exception("Error in set_voice callback (param)")
            elif param.name == "muted" and param.type_ == Parameter.Type.BOOL:
                command = "mute" if param.value else "unmute"
                logger.info("Parameter 'muted' changed to %s → sending %s", param.value, command)
                if self._command_callback is not None:
                    try:
                        self._command_callback(command)
                    except Exception:
                        logger.exception("Error in command callback (param)")

        return SetParametersResult(successful=True)

    # -- publishers ------------------------------------------------------------

    def publish_transcription(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_transcription.publish(msg)
        logger.debug("Published transcription: %s", text[:80])

    def publish_response(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_response.publish(msg)
        logger.debug("Published response: %s", text[:80])

    def publish_state(self, state: str) -> None:
        msg = String()
        msg.data = state
        self._pub_state.publish(msg)
        logger.debug("Published state: %s", state)

    def publish_audio_level(self, level: float) -> None:
        msg = Float32()
        msg.data = max(0.0, min(1.0, float(level)))
        self._pub_audio_level.publish(msg)

    def publish_metrics(self, metrics: dict) -> None:
        msg = String()
        msg.data = json.dumps(metrics)
        self._pub_metrics.publish(msg)
        logger.debug("Published metrics")

    def publish_bot_token(self, token: str) -> None:
        msg = String()
        msg.data = token
        self._pub_bot_token.publish(msg)

    def publish_bot_sentence(self, sentence: str) -> None:
        msg = String()
        msg.data = sentence
        self._pub_bot_sentence.publish(msg)
        logger.debug("Published bot sentence: %s", sentence[:80])

    def publish_wakeword(self, wakeword: str) -> None:
        msg = String()
        msg.data = wakeword
        self._pub_wakeword.publish(msg)
        logger.info("Published wakeword: %s", wakeword)

    def publish_info(self, info: dict) -> None:
        msg = String()
        msg.data = json.dumps(info)
        self._pub_info.publish(msg)
        logger.debug("Published info response")

    # -- callback setters ------------------------------------------------------

    def set_tts_callback(self, callback: Callable[[str], Any]) -> None:
        self._tts_callback = callback

    def set_command_callback(self, callback: Callable[[str], Any]) -> None:
        self._command_callback = callback

    def set_text_input_callback(self, callback: Callable[[str], Any]) -> None:
        self._text_input_callback = callback

    def set_interrupt_callback(self, callback: Callable[[], Any]) -> None:
        self._interrupt_callback = callback

    def set_set_language_callback(self, callback: Callable[[str], Any]) -> None:
        self._set_language_callback = callback

    def set_set_voice_callback(self, callback: Callable[[str], Any]) -> None:
        self._set_voice_callback = callback

    def set_query_callback(self, callback: Callable[[str], dict | None]) -> None:
        """Set callback for query commands (list_voices, list_languages, etc.).

        The callback receives the query name and returns a dict to publish on info topic.
        """
        self._query_callback = callback

    # -- subscriber handlers ---------------------------------------------------

    def _on_tts_request(self, msg: Any) -> None:
        text: str = msg.data
        logger.info("Received TTS request: %s", text[:80])
        if self._tts_callback is not None:
            try:
                self._tts_callback(text)
            except Exception:
                logger.exception("Error in TTS callback")

    def _on_command(self, msg: Any) -> None:
        command: str = msg.data
        logger.info("Received command: %s", command)

        # Handle query commands internally — publish response on info topic
        query_commands = {"list_voices", "list_languages", "hardware_info", "model_info"}
        if command in query_commands and self._query_callback is not None:
            try:
                result = self._query_callback(command)
                if result is not None:
                    self.publish_info({"query": command, **result})
            except Exception:
                logger.exception("Error in query callback for %s", command)
            return

        if self._command_callback is not None:
            try:
                self._command_callback(command)
            except Exception:
                logger.exception("Error in command callback")

    def _on_text_input(self, msg: Any) -> None:
        text: str = msg.data
        logger.info("Received text input: %s", text[:80])
        if self._text_input_callback is not None:
            try:
                self._text_input_callback(text)
            except Exception:
                logger.exception("Error in text input callback")

    def _on_interrupt(self, msg: Any) -> None:
        logger.info("Received interrupt request")
        if self._interrupt_callback is not None:
            try:
                self._interrupt_callback()
            except Exception:
                logger.exception("Error in interrupt callback")

    def _on_set_language(self, msg: Any) -> None:
        language: str = msg.data
        logger.info("Received set_language: %s", language)
        if self._set_language_callback is not None:
            try:
                self._set_language_callback(language)
            except Exception:
                logger.exception("Error in set_language callback")

    def _on_set_voice(self, msg: Any) -> None:
        voice: str = msg.data
        logger.info("Received set_voice: %s", voice)
        if self._set_voice_callback is not None:
            try:
                self._set_voice_callback(voice)
            except Exception:
                logger.exception("Error in set_voice callback")

    # -- lifecycle -------------------------------------------------------------

    def spin(self) -> None:
        """Start spinning the ROS2 node in a daemon thread."""
        if self._spin_thread is not None and self._spin_thread.is_alive():
            logger.warning("Spin thread already running")
            return

        def _spin_worker() -> None:
            logger.info("ROS2 spin thread started")
            try:
                while not self._shutdown_event.is_set() and rclpy.ok():
                    rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                logger.exception("ROS2 spin thread error")
            finally:
                logger.info("ROS2 spin thread exiting")

        self._spin_thread = threading.Thread(target=_spin_worker, daemon=True, name="ros2-spin")
        self._spin_thread.start()

    def shutdown(self) -> None:
        """Gracefully shut down the node and spin thread."""
        logger.info("Shutting down ROS2 bridge")
        self._shutdown_event.set()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None

        try:
            self._node.destroy_node()
        except Exception:
            logger.debug("Node already destroyed or rclpy shut down")

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            logger.debug("rclpy already shut down")


# ---------------------------------------------------------------------------
# NullBridge — no-op stand-in when ROS2 is unavailable or disabled
# ---------------------------------------------------------------------------
class NullBridge:
    """Drop-in replacement for ROS2Bridge that silently discards everything."""

    def __init__(self) -> None:
        logger.debug("Using NullBridge (ROS2 disabled or unavailable)")

    def publish_transcription(self, text: str) -> None: ...
    def publish_response(self, text: str) -> None: ...
    def publish_state(self, state: str) -> None: ...
    def publish_audio_level(self, level: float) -> None: ...
    def publish_metrics(self, metrics: dict) -> None: ...
    def publish_bot_token(self, token: str) -> None: ...
    def publish_bot_sentence(self, sentence: str) -> None: ...
    def publish_wakeword(self, wakeword: str) -> None: ...
    def publish_info(self, info: dict) -> None: ...

    def set_tts_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_command_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_text_input_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_interrupt_callback(self, callback: Callable[[], Any]) -> None: ...
    def set_set_language_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_set_voice_callback(self, callback: Callable[[str], Any]) -> None: ...
    def set_query_callback(self, callback: Callable[[str], dict | None]) -> None: ...

    def spin(self) -> None: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_bridge(enabled: bool = False, *, namespace: str = "/edgevox") -> ROS2Bridge | NullBridge:
    """Return a ROS2Bridge if *enabled* and rclpy is available, else NullBridge.

    Parameters
    ----------
    enabled:
        Set to ``True`` to attempt ROS2 initialisation.  When ``False``
        (the default) a lightweight :class:`NullBridge` is always returned.
    namespace:
        ROS2 namespace for the node.  Defaults to ``/edgevox``.  All topic
        names are relative, so the full topic path is ``<namespace>/<topic>``.
    """
    if not enabled:
        return NullBridge()

    if not ROS2_AVAILABLE:
        logger.warning("ROS2 bridge requested but rclpy is not installed — falling back to NullBridge")
        return NullBridge()

    try:
        bridge = ROS2Bridge(namespace=namespace)
        bridge.spin()
        return bridge
    except Exception:
        logger.exception("Failed to create ROS2 bridge — falling back to NullBridge")
        return NullBridge()
