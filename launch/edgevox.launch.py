"""ROS2 launch file for EdgeVox voice AI node.

Usage:
    ros2 launch edgevox edgevox.launch.py
    ros2 launch edgevox edgevox.launch.py namespace:=/robot1/voice language:=vi voice:=vi-vivos
"""

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="/edgevox", description="ROS2 namespace for the node"),
            DeclareLaunchArgument("language", default_value="en", description="Language ISO 639-1 code"),
            DeclareLaunchArgument("voice", default_value="", description="TTS voice name"),
            DeclareLaunchArgument("ui", default_value="simple", description="UI mode: simple, tui, or web"),
            DeclareLaunchArgument("domain_id", default_value="0", description="ROS_DOMAIN_ID"),
            ExecuteProcess(
                cmd=[
                    "edgevox",
                    "--ros2",
                    "--ros2-namespace",
                    LaunchConfiguration("namespace"),
                    "--language",
                    LaunchConfiguration("language"),
                    "--simple-ui",
                ],
                name="edgevox",
                output="screen",
                additional_env={"ROS_DOMAIN_ID": LaunchConfiguration("domain_id")},
            ),
        ]
    )
