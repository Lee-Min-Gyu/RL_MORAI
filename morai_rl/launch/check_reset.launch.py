from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("morai_rl"), "config", "stage1_rl8_ros2.toml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("repeats", default_value="5"),
            DeclareLaunchArgument("sleep_sec", default_value="0.5"),
            Node(
                package="morai_rl",
                executable="check_reset",
                name="morai_rl_check_reset",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("config"),
                    "--repeats",
                    LaunchConfiguration("repeats"),
                    "--sleep-sec",
                    LaunchConfiguration("sleep_sec"),
                ],
            ),
        ]
    )
