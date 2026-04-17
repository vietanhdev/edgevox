"""ROS2 robot-interop adapter — Nav2 / TF2 / sensor publishing.

Sits next to :class:`~edgevox.integrations.ros2_bridge.ROS2Bridge` and
bolts on the robotics-side plumbing that the voice bridge deliberately
stays out of:

- ``tf2_ros.TransformBroadcaster`` — publishes ``map → base_link`` (for
  2D mobile sims) or ``world → ee`` (for arm sims) at a configurable
  rate so rviz and Nav2 see the robot move in real time.
- ``geometry_msgs/PoseStamped`` on ``pose`` — the same pose as a regular
  topic for non-TF consumers.
- ``geometry_msgs/Twist`` subscriber on ``cmd_vel`` — Nav2-compatible
  velocity control, routed straight to ``apply_velocity(lin, ang)``.
- ``geometry_msgs/PoseStamped`` subscriber on ``goal_pose`` — RViz /
  Nav2's 2D goal pose, routed to ``apply_action("navigate_to", x, y)``
  for 2D mobile sims and ``apply_action("move_to", x, y, z)`` for arms.
- ``sensor_msgs/LaserScan`` on ``scan`` — 2D lidar publisher; sim
  must expose ``get_lidar_scan()``.
- ``sensor_msgs/Image`` on ``image_raw`` — offscreen camera publisher;
  sim must expose ``get_camera_frame(width, height)``.

The adapter auto-skips each capability that the underlying sim doesn't
provide — so a headless IR-SIM gets lidar + cmd_vel but not camera, a
MuJoCo arm gets image + goal_pose but not lidar, and a toy world gets
pose-only.

``rclpy`` and ``geometry_msgs`` / ``sensor_msgs`` / ``tf2_ros`` are
imported lazily so the module loads on non-ROS2 installs without
erroring — the public :func:`create_robot_adapter` returns ``None`` when
rclpy is missing.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rclpy  # noqa: F401 — import presence gates the rest of the module
    from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
    from sensor_msgs.msg import Image, LaserScan
    from tf2_ros import TransformBroadcaster

    from edgevox.integrations.ros2_qos import reliable_qos, sensor_qos

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


_DEFAULT_TF_HZ = 30.0
_DEFAULT_SCAN_HZ = 10.0
_DEFAULT_IMAGE_HZ = 5.0


class RobotROS2Adapter:
    """ROS2 robot-side publishers / subscribers attached to a sim.

    Shares the ``ROS2Bridge``'s ``rclpy.Node`` so everything lives under
    the same namespace. Owns its own publisher threads and cleans them
    up in :meth:`shutdown`.

    Args:
        node: the ROS2 node to attach to. Typically
            ``ROS2Bridge._node``.
        deps: the sim/robot environment. Duck-typed — the adapter
            enables each capability only if the relevant method exists.
            Known capabilities:

            - ``get_pose2d() -> (x, y, theta)``
            - ``get_ee_pose() -> (x, y, z)``
            - ``apply_velocity(linear, angular)``
            - ``apply_action("navigate_to", x=.., y=..) -> GoalHandle``
              (2D mobile)
            - ``apply_action("move_to", x, y, z) -> GoalHandle`` (arm)
            - ``get_lidar_scan() -> dict``
            - ``get_camera_frame(width, height) -> ndarray`` (HxWx3 uint8)

        base_frame: TF child frame for 2D mobile robots
            (default ``"base_link"``).
        ee_frame: TF child frame for arms (default ``"ee_link"``).
        map_frame: TF parent frame (default ``"map"``).
        tf_hz, scan_hz, image_hz: publisher rates.
        camera_size: ``(width, height)`` for the offscreen camera.
    """

    def __init__(
        self,
        node: Any,
        deps: Any,
        *,
        base_frame: str = "base_link",
        ee_frame: str = "ee_link",
        map_frame: str = "map",
        tf_hz: float = _DEFAULT_TF_HZ,
        scan_hz: float = _DEFAULT_SCAN_HZ,
        image_hz: float = _DEFAULT_IMAGE_HZ,
        camera_size: tuple[int, int] = (320, 240),
    ) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy / geometry_msgs / sensor_msgs not available")

        self._node = node
        self._deps = deps
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._map_frame = map_frame
        self._tf_hz = max(0.5, tf_hz)
        self._scan_hz = max(0.5, scan_hz)
        self._image_hz = max(0.5, image_hz)
        self._camera_size = camera_size

        self._has_pose2d = callable(getattr(deps, "get_pose2d", None))
        self._has_ee_pose = callable(getattr(deps, "get_ee_pose", None))
        self._has_velocity = callable(getattr(deps, "apply_velocity", None))
        self._has_apply_action = callable(getattr(deps, "apply_action", None))
        self._has_lidar = callable(getattr(deps, "get_lidar_scan", None))
        self._has_camera = callable(getattr(deps, "get_camera_frame", None))

        # Publishers --------------------------------------------------------
        self._tf_broadcaster = TransformBroadcaster(node)
        self._pose_pub = node.create_publisher(PoseStamped, "pose", reliable_qos())
        self._scan_pub = node.create_publisher(LaserScan, "scan", sensor_qos()) if self._has_lidar else None
        self._image_pub = node.create_publisher(Image, "image_raw", sensor_qos()) if self._has_camera else None

        # Subscribers -------------------------------------------------------
        self._cmd_vel_sub = (
            node.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, reliable_qos()) if self._has_velocity else None
        )
        self._goal_pose_sub = node.create_subscription(PoseStamped, "goal_pose", self._on_goal_pose, reliable_qos())

        # Background loops --------------------------------------------------
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        if self._has_pose2d or self._has_ee_pose:
            t = threading.Thread(target=self._tf_pose_loop, name="ros2-tf-pose", daemon=True)
            t.start()
            self._threads.append(t)
        if self._scan_pub is not None:
            t = threading.Thread(target=self._scan_loop, name="ros2-scan", daemon=True)
            t.start()
            self._threads.append(t)
        if self._image_pub is not None:
            t = threading.Thread(target=self._image_loop, name="ros2-image", daemon=True)
            t.start()
            self._threads.append(t)

        logger.info(
            "RobotROS2Adapter up — pose2d=%s ee_pose=%s vel=%s lidar=%s camera=%s",
            self._has_pose2d,
            self._has_ee_pose,
            self._has_velocity,
            self._has_lidar,
            self._has_camera,
        )

    # ----- shutdown --------------------------------------------------------

    def shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.5)
        self._threads.clear()

    # ----- time helper -----------------------------------------------------

    def _now_stamp(self) -> Any:
        return self._node.get_clock().now().to_msg()

    # ----- TF2 + PoseStamped ----------------------------------------------

    def _tf_pose_loop(self) -> None:
        period = 1.0 / self._tf_hz
        while not self._stop.wait(period):
            try:
                if self._has_pose2d:
                    self._publish_pose2d()
                elif self._has_ee_pose:
                    self._publish_ee_pose()
            except Exception:
                logger.debug("tf/pose publish failed", exc_info=True)

    def _publish_pose2d(self) -> None:
        x, y, theta = self._deps.get_pose2d()
        stamp = self._now_stamp()

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._base_frame
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = 0.0
        qz = math.sin(theta * 0.5)
        qw = math.cos(theta * 0.5)
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self._map_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self._pose_pub.publish(ps)

    def _publish_ee_pose(self) -> None:
        x, y, z = self._deps.get_ee_pose()
        stamp = self._now_stamp()

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._map_frame
        tf.child_frame_id = self._ee_frame
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        tf.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self._map_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.w = 1.0
        self._pose_pub.publish(ps)

    # ----- LaserScan -------------------------------------------------------

    def _scan_loop(self) -> None:
        period = 1.0 / self._scan_hz
        while not self._stop.wait(period):
            try:
                scan = self._deps.get_lidar_scan()
            except Exception:
                logger.debug("get_lidar_scan failed", exc_info=True)
                continue
            if not scan:
                continue
            try:
                self._publish_scan(scan)
            except Exception:
                logger.debug("LaserScan publish failed", exc_info=True)

    def _publish_scan(self, scan: dict[str, Any]) -> None:
        msg = LaserScan()
        msg.header.stamp = self._now_stamp()
        msg.header.frame_id = self._base_frame
        msg.angle_min = float(scan.get("angle_min", -math.pi))
        msg.angle_max = float(scan.get("angle_max", math.pi))
        msg.angle_increment = float(scan.get("angle_increment", scan.get("angle_inc", 0.0)))
        msg.time_increment = float(scan.get("time_increment", scan.get("time_inc", 0.0)))
        msg.scan_time = float(scan.get("scan_time", 1.0 / self._scan_hz))
        msg.range_min = float(scan.get("range_min", 0.0))
        msg.range_max = float(scan.get("range_max", 10.0))
        ranges = scan.get("ranges")
        if ranges is None:
            return
        try:
            msg.ranges = [float(r) for r in ranges]
        except TypeError:
            msg.ranges = list(map(float, list(ranges)))
        intensities = scan.get("intensities")
        if intensities is not None:
            try:
                msg.intensities = [float(i) for i in intensities]
            except Exception:
                msg.intensities = []
        self._scan_pub.publish(msg)

    # ----- Image -----------------------------------------------------------

    def _image_loop(self) -> None:
        period = 1.0 / self._image_hz
        w, h = self._camera_size
        while not self._stop.wait(period):
            try:
                frame = self._deps.get_camera_frame(width=w, height=h)
            except Exception:
                logger.debug("get_camera_frame failed", exc_info=True)
                continue
            if frame is None:
                continue
            try:
                self._publish_image(frame)
            except Exception:
                logger.debug("Image publish failed", exc_info=True)

    def _publish_image(self, frame: Any) -> None:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w = arr.shape[:2]
        msg = Image()
        msg.header.stamp = self._now_stamp()
        msg.header.frame_id = f"{self._ee_frame}_camera" if self._has_ee_pose else f"{self._base_frame}_camera"
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(w * 3)
        msg.data = arr.astype("uint8").tobytes()
        self._image_pub.publish(msg)

    # ----- subscriber callbacks -------------------------------------------

    def _on_cmd_vel(self, msg: Any) -> None:
        if not self._has_velocity:
            return
        try:
            self._deps.apply_velocity(float(msg.linear.x), float(msg.angular.z))
        except Exception:
            logger.exception("cmd_vel dispatch failed")

    def _on_goal_pose(self, msg: Any) -> None:
        if not self._has_apply_action:
            return
        try:
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
            z = float(msg.pose.position.z)
        except Exception:
            logger.debug("bad goal_pose message", exc_info=True)
            return

        if self._has_ee_pose:
            with contextlib.suppress(Exception):
                self._deps.apply_action("move_to", x=x, y=y, z=z)
            return
        if self._has_pose2d:
            with contextlib.suppress(Exception):
                self._deps.apply_action("navigate_to", x=x, y=y)
            return


def create_robot_adapter(
    node: Any,
    deps: Any,
    **kwargs: Any,
) -> RobotROS2Adapter | None:
    """Return a :class:`RobotROS2Adapter` if rclpy is available and
    ``deps`` exposes at least one supported capability; else ``None``.
    """
    if not ROS2_AVAILABLE or node is None or deps is None:
        return None
    if not any(
        callable(getattr(deps, m, None))
        for m in ("get_pose2d", "get_ee_pose", "get_lidar_scan", "get_camera_frame", "apply_velocity")
    ):
        return None
    try:
        return RobotROS2Adapter(node, deps, **kwargs)
    except Exception:
        logger.exception("RobotROS2Adapter init failed")
        return None


__all__ = ["RobotROS2Adapter", "create_robot_adapter"]
