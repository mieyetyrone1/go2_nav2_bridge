#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import urllib.request
import os

class YoloONNXNode(Node):
    def __init__(self):
        super().__init__('yolo_bytetrack_node')
        
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.bbox_pub = self.create_publisher(PolygonStamped, '/tracker/tracked_target', 10)
        self.bridge = CvBridge()
        
        # 1. Download an optimized, lightweight YOLOv8s ONNX model file
        self.model_path = os.path.expanduser('~/yolov8s.onnx')
        if not os.path.exists(self.model_path):
            self.get_logger().info('Downloading lightweight, resource-optimized YOLO ONNX model...')
            url = 'https://github.com'
            urllib.request.urlretrieve(url, self.model_path)
            
        # 2. Load the network cleanly using OpenCV's DNN engine (No pickle errors!)
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.get_logger().info('Resource-optimized ONNX model loaded successfully!')
        
        # 3. Target class index (In standard COCO models, Backpack is class index 24)
        self.target_class_id = 24 
        self.confidence_threshold = 0.4

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        h, w, _ = cv_image.shape
        
        # Prepare image blob for the neural network
        blob = cv2.dnn.blobFromImage(cv_image, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Parse standard YOLO outputs: [1, 84, 8400] -> [8400, 84]
        predictions = np.squeeze(outputs).T
        
        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id == self.target_class_id and confidence > self.confidence_threshold:
                # Extract box coordinates (normalized center_x, center_y, width, height)
                cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
                
                # Convert back to raw image pixels
                x_min = int((cx - bw/2) * w)
                y_min = int((cy - bh/2) * h)
                x_max = int((cx + bw/2) * w)
                y_max = int((cy + bh/2) * h)
                
                # Package and publish data to your C++ node
                bbox_msg = PolygonStamped()
                bbox_msg.header.stamp = msg.header.stamp
                bbox_msg.header.frame_id = 'camera_depth_optical_frame'
                
                p1 = Point32(x=float(x_min), y=float(y_min), z=0.0)
                p2 = Point32(x=float(x_max), y=float(y_max), z=0.0)
                bbox_msg.polygon.points = [p1, p2]
                
                self.bbox_pub.publish(bbox_msg)
                break # Track the main backpack to keep compute high

def main(args=None):
    rclpy.init(args=args)
    node = YoloONNXNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

