import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_bridge = get_package_share_directory('go2_nav2_bridge')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    params_file = os.path.join(pkg_bridge, 'config', 'nav2_params.yaml')

    # 1. Custom Unified Movement & Transform Bridge
    nav2_bridge_node = Node(
        package='go2_nav2_bridge',
        executable='nav2_bridge_node',
        name='go2_nav2_bridge',
        output='screen'
    )

    # 2. Nav2 Stack Launch Core
    nav2_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'False',
            'params_file': params_file,
            'autostart': 'True'
        }.items()
    )

    return LaunchDescription([
        nav2_bridge_node,
        nav2_stack
    ])

