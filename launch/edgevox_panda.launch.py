"""ROS2 launch file for the MuJoCo Panda voice agent demo.

Starts ``edgevox-agent robot-panda --text-mode --ros2`` under the
requested namespace. Publishes ``robot_state`` with end-effector pose
and currently-grasped object; ``agent_event`` streams skill + tool
events; ``text_input`` drives pick-and-place commands.

Usage::

    ros2 launch edgevox edgevox_panda.launch.py
    ros2 launch edgevox edgevox_panda.launch.py namespace:=/robot1/arm

    # From another terminal:
    ros2 topic pub --once /edgevox/text_input std_msgs/msg/String \
      "{data: 'pick up the red cube'}"
    ros2 topic echo /edgevox/robot_state --once
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
                "gantry",
                default_value="false",
                description="Use the bundled primitive gantry scene (zero network) instead of Franka",
            ),
            ExecuteProcess(
                cmd=[
                    "edgevox-agent",
                    "robot-panda",
                    "--text-mode",
                    "--ros2",
                    "--ros2-namespace",
                    LaunchConfiguration("namespace"),
                    "--ros2-state-hz",
                    LaunchConfiguration("state_hz"),
                    "--no-render",
                ],
                name="edgevox_panda",
                output="screen",
                additional_env={"ROS_DOMAIN_ID": LaunchConfiguration("domain_id")},
            ),
        ]
    )
