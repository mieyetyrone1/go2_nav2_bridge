#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PolygonStamped, Point32
from sensor_msgs.msg import Image, CameraInfo
import numpy as np

class MockTrackerNode(Node):
    def __init__(self):
        super().__init__('mock_tracker_node')
        self.bbox_pub = self.create_publisher(PolygonStamped, '/tracker/tracked_target', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Mock Tracker & Depth stream initialized at 10Hz!')

    def timer_callback(self):
        # Fetch the exact time snapshot once
        current_time = self.get_clock().now().to_msg()

        # 1. Camera Info
        info_msg = CameraInfo()
        info_msg.header.stamp = current_time
        info_msg.header.frame_id = 'camera_depth_optical_frame'
        info_msg.width = 640
        info_msg.height = 480
        info_msg.k = [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info_msg)

        # 2. Bounding Box
        bbox_msg = PolygonStamped()
        bbox_msg.header.stamp = current_time
        bbox_msg.header.frame_id = 'camera_depth_optical_frame'
        p1 = Point32(x=310.0, y=230.0, z=0.0)
        p2 = Point32(x=330.0, y=250.0, z=0.0)
        bbox_msg.polygon.points = [p1, p2]
        self.bbox_pub.publish(bbox_msg)

        # 3. Depth Image
        depth_msg = Image()
        depth_msg.header.stamp = current_time
        depth_msg.header.frame_id = 'camera_depth_optical_frame'
        depth_msg.height = 480
        depth_msg.width = 640
        depth_msg.encoding = '16UC1'
        depth_msg.is_bigendian = 0
        depth_msg.step = 640 * 2
        
        depth_array = np.full((480, 640), 1000, dtype=np.uint16)
        depth_msg.data = depth_array.tobytes()
        self.depth_pub.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MockTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
