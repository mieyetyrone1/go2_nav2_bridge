import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # External package directory for Nav2 core tracking logic
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Absolute path directly to src/ allows instant parameter adjustments without rebuilding
    params_file = '/home/unitree/go2_nav2_ws/src/go2_nav2_bridge/config/nav2_params.yaml'

    # Create a configurable runtime argument (defaults to INFO for production)
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal)'
    )

    # 1. Hardware Bridge node (Locomotion control, pointcloud processing, and network time mapping)
    nav2_bridge_node = Node(
        package='go2_nav2_bridge',
        executable='nav2_bridge_node',
        name='go2_nav2_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'transform_tolerance': 0.5
        }],

        # Dynamically bind the runtime arguments here
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # 2. Static Transform Publisher (Establishes physical coordinate link between body center and LiDAR sensor)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.25', '0.0', '0.15', '0', '0', '0', 'base_link', 'utlidar_lidar'],
        output='screen'
    )

    # 3. Nav2 Stack Launch Core (Fires up the costmaps and RPP/NavFn planner nodes)
    nav2_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'False',
            'params_file': params_file,
            'autostart': 'True'
        }.items()
    )

    return LaunchDescription([
        log_level_arg,
        nav2_bridge_node,
        static_tf_node,
        nav2_stack
    ])

