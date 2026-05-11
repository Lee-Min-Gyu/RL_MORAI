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
            DeclareLaunchArgument("episodes", default_value="5"),
            DeclareLaunchArgument("steps", default_value="60"),
            DeclareLaunchArgument("throttle", default_value="0.25"),
            DeclareLaunchArgument("steering", default_value="0.0"),
            Node(
                package="morai_rl",
                executable="check_reset_drive",
                name="morai_rl_check_reset_drive",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("config"),
                    "--episodes",
                    LaunchConfiguration("episodes"),
                    "--steps",
                    LaunchConfiguration("steps"),
                    "--throttle",
                    LaunchConfiguration("throttle"),
                    "--steering",
                    LaunchConfiguration("steering"),
                ],
            ),
        ]
    )
