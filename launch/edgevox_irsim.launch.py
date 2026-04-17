"""ROS2 launch file for the IR-SIM voice agent demo.

Starts ``edgevox-agent robot-irsim --text-mode --ros2`` under the
requested namespace. The agent publishes ``robot_state``,
``agent_event``, ``transcription``, and ``response`` on
``<namespace>/...``. Commands flow in on ``<namespace>/text_input``.

Usage::

    ros2 launch edgevox edgevox_irsim.launch.py
    ros2 launch edgevox edgevox_irsim.launch.py namespace:=/robot1/voice

    # From another terminal, drive the robot:
    ros2 topic pub --once /edgevox/text_input std_msgs/msg/String \
      "{data: 'go to the kitchen'}"
"""

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="/edgevox", description="ROS2 namespace for the bridge"),
            DeclareLaunchArgument(
                "state_hz", default_value="10.0", description="Rate at which robot_state is republished"
            ),
            DeclareLaunchArgument("domain_id", default_value="0", description="ROS_DOMAIN_ID"),
            DeclareLaunchArgument(
                "render",
                default_value="false",
                description="Open the matplotlib IR-SIM window (needs an X display)",
            ),
            ExecuteProcess(
                cmd=[
                    "edgevox-agent",
                    "robot-irsim",
                    "--text-mode",
                    "--ros2",
                    "--ros2-namespace",
                    LaunchConfiguration("namespace"),
                    "--ros2-state-hz",
                    LaunchConfiguration("state_hz"),
                    # Headless by default — override to true for the viewer
                    # (handled upstream via ``--no-render``).
                ],
                name="edgevox_irsim",
                output="screen",
                additional_env={"ROS_DOMAIN_ID": LaunchConfiguration("domain_id")},
            ),
        ]
    )
