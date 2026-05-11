import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'morai_standard'
    pkg_share_dir = get_package_share_directory(pkg_name)
    
    rviz_config_path = os.path.join(pkg_share_dir, 'rviz', 'morai_standard.rviz')

    morai_node = Node(
        package=pkg_name,
        executable='morai_standard_node',
        name='morai_standard',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription([
        morai_node,
        rviz_node
    ])