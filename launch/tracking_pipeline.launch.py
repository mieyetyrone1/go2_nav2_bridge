from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Physical Structural Transform (Nose Mount)
    # Connects Go2 body axis to RealSense physical chassis center
    static_tf_structural = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='nose_mount_publisher',
        arguments=['0.25', '0.0', '0.05', '0', '0', '0', 'base_link', 'camera_link']
    )

    # Camera Optical Frame Transform 
    # Aligns the structural frame to the image standard (Z forward, X right, Y down)
    static_tf_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_publisher',
        arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'camera_link', 'camera_depth_optical_frame']
    )

    # Depth Image to LaserScan Node
    # Compresses raw depth image slices into a 2D flat barrier map for active obstacle avoidance
    depth_to_laserscan_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        output='screen',
        remappings=[
            ('depth', '/camera/depth/image_rect_raw'),
            ('depth_camera_info', '/camera/depth/camera_info'),
            ('scan', '/camera_scan') 
        ],
        parameters=[{
            'scan_time': 0.033,
            'scan_height': 20,
            'depth_min': 0.25,
            'depth_max': 3.5,
            'output_frame_id': 'camera_depth_optical_frame' # Keep this here
        }],
        arguments=['--ros-args', '-p', 'output_frame_id:=camera_depth_optical_frame']
    )

    # Target Projector C++ Node
    target_projector_node = Node(
        package='go2_nav2_bridge',
        executable='target_projector_node',
        name='target_projector',
        output='screen'
    )

    return LaunchDescription([
        static_tf_structural,
        static_tf_optical,
        depth_to_laserscan_node,
        target_projector_node
    ])

