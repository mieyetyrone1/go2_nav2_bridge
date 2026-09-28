#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/polygon_stamped.hpp" 

#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

class TargetProjectorNode : public rclcpp::Node
{
public:
    TargetProjectorNode() : Node("target_projector_node")
    {
        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Subscribe to RealSense calibration parameters stream
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/depth/camera_info", 10,
            std::bind(&TargetProjectorNode::infoCallback, this, std::placeholders::_1));

        bbox_sub_.subscribe(this, "/tracker/bbox");
        depth_sub_.subscribe(this, "/camera/depth/image_rect_raw"); // RealSense convention topic

        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), bbox_sub_, depth_sub_);
            
        sync_->registerCallback(std::bind(&TargetProjectorNode::synchronizedCallback, 
            this, std::placeholders::_1, std::placeholders::_2));

        has_calibration_ = false;
        RCLCPP_INFO(this->get_logger(), "C++ Dynamic Target Projector Node Initialized!");
    }

private:
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!has_calibration_) {
            // Parse K matrix values: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            fx_ = msg->k[0];
            cx_ = msg->k[2];
            fy_ = msg->k[4];
            cy_ = msg->k[5];
            has_calibration_ = true;
            RCLCPP_INFO(this->get_logger(), "Successfully loaded real-time RealSense calibration matrix components.");
        }
    }

    void synchronizedCallback(
        const geometry_msgs::msg::PolygonStamped::ConstSharedPtr bbox_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr depth_msg)
    {
        if (!has_calibration_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for initial CameraInfo metadata stream...");
            return;
        }

        if (bbox_msg->polygon.points.size() < 2) { 
            return; 
        }

        double x_min = bbox_msg->polygon.points[0].x;
        double y_min = bbox_msg->polygon.points[0].y;
        double x_max = bbox_msg->polygon.points[1].x;
        double y_max = bbox_msg->polygon.points[1].y;

        int u = static_cast<int>(x_min + (x_max - x_min) / 2.0);
        int v = static_cast<int>(y_min + (y_max - y_min) / 2.0);

        if (u < 0 || u >= static_cast<int>(depth_msg->width) || v < 0 || v >= static_cast<int>(depth_msg->height)) {
            return;
        }

        // RealSense standard raw depth format is uint16_t in millimeters
        const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
        int row_stride = depth_msg->step / sizeof(uint16_t);
        double z_c = depth_data[v * row_stride + u] / 1000.0; 

        if (z_c <= 0.1 || z_c > 4.0) { // Prune noise limits outside of steady tracking zones
            return;
        }

        geometry_msgs::msg::PointStamped camera_point;
        camera_point.header = depth_msg->header; 
        camera_point.point.x = ((u - cx_) * z_c) / fx_;
        camera_point.point.y = ((v - cy_) * z_c) / fy_;
        camera_point.point.z = z_c;

        geometry_msgs::msg::PointStamped odom_point;
        try {
            odom_point = tf_buffer_->transform(camera_point, "odom");
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform execution failed: %s", ex.what());
            return;
        }

        geometry_msgs::msg::PoseStamped nav_goal;
        nav_goal.header.stamp = depth_msg->header.stamp;
        nav_goal.header.frame_id = "odom"; 
        nav_goal.pose.position = odom_point.point;
        nav_goal.pose.orientation.w = 1.0; 

        goal_pub_->publish(nav_goal);
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    message_filters::Subscriber<geometry_msgs::msg::PolygonStamped> bbox_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;

    using SyncPolicy = message_filters::sync_policies::ExactTime<geometry_msgs::msg::PolygonStamped, sensor_msgs::msg::Image>;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    double fx_, fy_, cx_, cy_;
    bool has_calibration_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    std::shared_ptr<TargetProjectorNode> node = std::make_shared<TargetProjectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

