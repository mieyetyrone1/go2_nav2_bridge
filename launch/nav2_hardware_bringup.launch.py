import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    nav2_bridge_node = Node(
        package='go2_nav2_bridge',
        executable='nav2_bridge_node',
        name='go2_nav2_bridge',
        output='screen'
    )

    base_link_to_lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_laser',
        arguments=['0.25', '0.0', '0.15', '0.0', '0.0', '0.0', 'base_link', 'utlidar_link']
    )

    return LaunchDescription([
        nav2_bridge_node,
        base_link_to_lidar_tf
    ])

