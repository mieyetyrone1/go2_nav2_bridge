#!/usr/bin/env python3

# ==============================================================================
# CRITICAL JETSON TLS MEMORY WORKAROUND
# ==============================================================================
import torch  # Must remain first to force static TLS slot assignment
# ==============================================================================

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2

# Import the dedicated Open-Vocabulary YOLO-World Engine
from ultralytics import YOLOWorld

class YoloByteTrackNode(Node):
    def __init__(self):
        super().__init__('yolo_bytetrack_processor')
        
        # Topic Subscriptions & Publishers
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.camera_callback, 10)
        self.bbox_pub = self.create_publisher(PolygonStamped, '/tracker/tracked_target', 10)
        self.bridge = CvBridge()
        
        # Load Dedicated YOLO-World Model
        model_path = os.path.expanduser('~/go2_nav2_ws/yolov8s-worldv2.pt')
        self.get_logger().info(f"Loading YOLO-World open-vocabulary engine from: {model_path}")
        self.model = YOLOWorld(model_path)
        
        # CONFIGURE CUSTOM OPEN-VOCABULARY TEXT TARGET
        self.target_class_label = "brown box" 
        self.model.set_classes([self.target_class_label])
        
        self.get_logger().info(f"YOLO-World + ByteTrack initialized! Tracking custom text prompt: '{self.target_class_label}'")

    def camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Execute Tracking Loop using the custom text vocabulary profile
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                conf=0.20, 
                verbose=False
            )
            
            # Check if targets are active in frame
            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                # Extract raw numpy arrays correctly from the results object wrapper
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                if len(boxes) > 0:
                    # Explicitly grab the first row index [0] to keep array dimensions 1D
                    x1, y1, x2, y2 = boxes[0]
                    assigned_id = track_ids[0]
                    
                    bbox_msg = PolygonStamped()
                    bbox_msg.header.stamp = msg.header.stamp  
                    bbox_msg.header.frame_id = "camera_depth_optical_frame"
                    
                    p1 = Point32(x=float(x1), y=float(y1), z=float(assigned_id))
                    p2 = Point32(x=float(x2), y=float(y2), z=float(assigned_id))
                    bbox_msg.polygon.points = [p1, p2]
                    
                    self.bbox_pub.publish(bbox_msg)
                    self.get_logger().info(
                        f"[YOLO-WORLD LOCK] Matched text prompt '{self.target_class_label}' ID #{assigned_id} at Box: [{int(x1)}, {int(y1)}] -> [{int(x2)}, {int(y2)}]"
                    )
                    return
            
            self.get_logger().info(f"Scanning... Custom prompt '{self.target_class_label}' not detected.", throttle_duration_sec=2.0)
            
        except Exception as e:
            self.get_logger().error(f"YOLO-World pipeline tracking failure: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloByteTrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

