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
            DeclareLaunchArgument("timesteps", default_value="50000"),
            DeclareLaunchArgument("run_name", default_value="ros2_morai_rl"),
            Node(
                package="morai_rl",
                executable="train_ppo",
                name="morai_rl_train_ppo",
                output="screen",
                arguments=[
                    "--config",
                    LaunchConfiguration("config"),
                    "--timesteps",
                    LaunchConfiguration("timesteps"),
                    "--run-name",
                    LaunchConfiguration("run_name"),
                ],
            ),
        ]
    )
