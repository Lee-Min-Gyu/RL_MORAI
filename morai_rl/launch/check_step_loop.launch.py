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
            DeclareLaunchArgument("steps", default_value="50"),
            DeclareLaunchArgument("throttle", default_value="0.0"),
            DeclareLaunchArgument("brake", default_value="0.0"),
            DeclareLaunchArgument("steering", default_value="0.0"),
            Node(
                package="morai_rl",
                executable="check_step_loop",
                name="morai_rl_check_step_loop",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("config"),
                    "--steps",
                    LaunchConfiguration("steps"),
                    "--throttle",
                    LaunchConfiguration("throttle"),
                    "--brake",
                    LaunchConfiguration("brake"),
                    "--steering",
                    LaunchConfiguration("steering"),
                ],
            ),
        ]
    )
