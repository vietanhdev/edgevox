"""Shared ROS2 QoS profiles for EdgeVox publishers / subscribers / tests.

Before this module, three different files carried their own copies of
the same three profiles (state / sensor / reliable). Centralising them
here keeps behaviour consistent: if we change the history depth or
durability for, say, the sensor profile, every publisher and every
test updates at once.

Import lazily — ``rclpy`` isn't a pip package, so machines without a
sourced ROS2 workspace need the module to load without raising.
:func:`state_qos` and friends raise ``RuntimeError`` when rclpy is
missing; callers that live behind a ``ROS2_AVAILABLE`` gate never
reach them.
"""

from __future__ import annotations

from typing import Any

try:
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False


def _guard() -> None:
    if not _RCLPY_AVAILABLE:
        raise RuntimeError("rclpy is not installed — QoS profiles are only available with a sourced ROS2 workspace")


def state_qos() -> Any:
    """State-topic profile: RELIABLE + TRANSIENT_LOCAL + depth 1.

    Late-joining subscribers see the last published value immediately
    — used for pipeline/robot state snapshots where the current value
    is far more useful than the history.
    """
    _guard()
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def sensor_qos() -> Any:
    """High-frequency sensor profile: BEST_EFFORT + VOLATILE + depth 5.

    Drops stale samples when the subscriber can't keep up rather than
    accumulating latency. Used for audio levels, streaming tokens,
    LaserScan, Image.
    """
    _guard()
    return QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )


def reliable_qos() -> Any:
    """Event / command profile: RELIABLE + VOLATILE + depth 10.

    Messages must not be lost (commands, events, transcriptions)
    but late joiners don't replay history.
    """
    _guard()
    return QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )


__all__ = ["reliable_qos", "sensor_qos", "state_qos"]
