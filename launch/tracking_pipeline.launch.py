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
        target_projector_node
    ])

