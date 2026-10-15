import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('go2_nav2_bridge')
    
    # Locate the newly created SLAM parameter asset file path
    slam_config_path = os.path.join(pkg_share, 'config', 'mapper_params_online_async.yaml')

    # 1. PointCloud to LaserScan Projection Node Configuration
    # Slices the 3D points at the Go2's shoulder height to generate a clear 2D costmap line
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/utlidar/cloud_deskewed'),
            ('scan', '/scan')
        ],
        parameters=[{
            'target_frame': 'utlidar_lidar', # Lock projection center to the sensor's physical center
            'transform_tolerance': 0.01,
            'min_height': -0.3,             # Prune floor reflections below the feet to avoid ghost walls
            'max_height': 0.5,              # Capture obstacles up to head height
            'angle_min': -3.14159,          # Complete 360-degree horizontal field of view wrapping
            'angle_max': 3.14159,
            'angle_increment': 0.0087,      # 0.5 degree scanning beam resolution
            'scan_time': 0.1,
            'range_min': 0.2,
            'range_max': 25.0,
            'use_inf': True
        }],
        output='screen'
    )

    # 2. Main Graph-Based SLAM Toolbox Asynchronous Mapping Node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_config_path],
        output='screen'
    )

    return LaunchDescription([
        pointcloud_to_laserscan_node,
        slam_toolbox_node
    ])

