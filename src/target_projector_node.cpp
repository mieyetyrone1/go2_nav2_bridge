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
#include "message_filters/sync_policies/approximate_time.h"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

class TargetProjectorNode : public rclcpp::Node
{
public:
    TargetProjectorNode() : Node("target_projector_node")
    {
        // Declare the Sensor Data QoS Profile to handle the RealSense driver rules
        rmw_qos_profile_t custom_qos = rmw_qos_profile_sensor_data;
        custom_qos.depth = 10;

        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);
        
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/depth/camera_info", 10,
            std::bind(&TargetProjectorNode::infoCallback, this, std::placeholders::_1));
        
        bbox_sub_.subscribe(this, "/tracker/tracked_target", custom_qos);
        depth_sub_.subscribe(this, "/camera/depth/image_rect_raw", custom_qos);
        
        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(30), bbox_sub_, depth_sub_);
        
        sync_->setInterMessageLowerBound(0, rclcpp::Duration::from_seconds(0.1));
        sync_->setInterMessageLowerBound(1, rclcpp::Duration::from_seconds(0.1));

        sync_->registerCallback(std::bind(&TargetProjectorNode::synchronizedCallback,
            this, std::placeholders::_1, std::placeholders::_2));
        
        has_calibration_ = false;
        has_sent_first_goal_ = false;
        last_goal_published_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "C++ Dynamic Target Projector Node Initialized!");
    }

private:
    /**
     * @brief Parses and extracts intrinsic camera calibration matrices.
     * 
     * Listens to the projection metadata stream to capture horizontal and vertical
     * focal lengths alongside principal point configurations necessary to perform 
     * precise 2D-to-3D optical matrix calculations.
     * 
     * @param msg Shared pointer to the incoming sensor_msgs::msg::CameraInfo stream.
     */
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

    /**
     * @brief Synchronizes object tracking bounding boxes with raw depth image matrices.
     * 
     * Extracts target bounding coordinates via exact time matching, samples focal center point
     * depth values (uint16_t millimeters), and constructs ray-traced spatial point vectors. 
     * Coordinates are converted directly into the 'odom' spatial frame and broadcast to Nav2.
     * 
     * @param bbox_msg Time-synchronized polygon representing tracking boundary corners.
     * @param depth_msg Time-synchronized raw rectified optical depth register packet.
     */
    void synchronizedCallback(
        const geometry_msgs::msg::PolygonStamped::ConstSharedPtr bbox_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr depth_msg)
    {
        // 1. TIME-GATE: Throttle updates to a maximum frequency of 2 Hz (0.5 second interval)
        rclcpp::Time current_time = this->now();
        double time_since_last_goal = (current_time - last_goal_published_time_).seconds();

        if (time_since_last_goal < 0.5 && time_since_last_goal > 0.0) {
            // Drop the callback silently to keep the pipeline moving without flooding Nav2
            return;
        }

        if (!has_calibration_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for initial CameraInfo metadata stream...");
            return;
        }

        if (bbox_msg->polygon.points.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "[DEBUG FAIL] Polygon box message has less than 2 points!");
            return;
        }

        // Extract the bounding box coordinates (assuming standard bounding box format)
        double x_min = bbox_msg->polygon.points[0].x;
        double y_min = bbox_msg->polygon.points[0].y;
        double x_max = bbox_msg->polygon.points[1].x;
        double y_max = bbox_msg->polygon.points[1].y;

        int u = static_cast<int>(x_min + (x_max - x_min) / 2.0);
        int v = static_cast<int>(y_min + (y_max - y_min) / 2.0);

        if (u < 0 || u >= static_cast<int>(depth_msg->width) || v < 0 || v >= static_cast<int>(depth_msg->height)) {
            RCLCPP_WARN(this->get_logger(), "[DEBUG FAIL] Calculated center pixel falls OUTSIDE depth image dimensions!");
            return;
        }

        // Read raw depth (RealSense format: uint16_t in millimeters)
        const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
        int row_stride = depth_msg->step / sizeof(uint16_t);
        double z_c = depth_data[v * row_stride + u] / 1000.0;

        // Depth constraints floor and ceiling gates
        if (z_c <= 0.1 || z_c > 6.0) { 
            RCLCPP_WARN(this->get_logger(), "[DEBUG FAIL] Distance %f is outside of tracking constraints (0.1m - 6.0m)!", z_c);
            return;
        }

        // Apply the 0.5m safety braking vector shift
        const double SAFETY_BUFFER = 0.5; 
        if (z_c > SAFETY_BUFFER) {
            z_c = z_c - SAFETY_BUFFER; 
        } else {
            z_c = 0.0;
        }

        // Project raw 3D camera frame coordinates
        geometry_msgs::msg::PointStamped camera_point;

        // FORCE lookups to use the current time clock frame to eliminate history warping
        camera_point.header.stamp = rclcpp::Time(0, 0); 
        camera_point.header.frame_id = depth_msg->header.frame_id;
        camera_point.point.x = ((u - cx_) * z_c) / fx_;
        camera_point.point.y = ((v - cy_) * z_c) / fy_;
        camera_point.point.z = z_c;

        geometry_msgs::msg::PointStamped odom_point;
        try {
            // Enforce a strict transform lookup gate
            odom_point = tf_buffer_->transform(camera_point, "odom");
        }
        catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform execution failed: %s", ex.what());
            return;
        }

        // --- DISPLACEMENT ARRIVAL GATE ---
        if (has_sent_first_goal_) {
            double dx = odom_point.point.x - last_sent_odom_point_.x;
            double dy = odom_point.point.y - last_sent_odom_point_.y;
            double target_displacement = std::sqrt(dx * dx + dy * dy);

            // If the target calculation shifts less than 15cm, drop the frame 
            // and allow Nav2 to execute its stable arrival braking trajectory.
            if (target_displacement < 0.15) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                    "[SHADOW] Target stationary (shifted %.2fm). Allowing Nav2 to decelerate.", target_displacement);
                return; 
            }
        }

        // Construct the full Nav2 Goal message
        geometry_msgs::msg::PoseStamped nav_goal;
        nav_goal.header.stamp = this->now(); // Clear historical stamp latency
        nav_goal.header.frame_id = "odom";
        nav_goal.pose.position = odom_point.point;
        nav_goal.pose.orientation.x = 0.0;
        nav_goal.pose.orientation.y = 0.0;
        nav_goal.pose.orientation.z = 0.0;
        nav_goal.pose.orientation.w = 1.0;

        RCLCPP_INFO(this->get_logger(), "[SHADOW UPDATE] Streaming continuous Odom target at X: %f, Y: %f (dt: %.2fs)", 
                    odom_point.point.x, odom_point.point.y, time_since_last_goal);
        
        // Publish and cache states
        goal_pub_->publish(nav_goal);
        last_sent_odom_point_ = odom_point.point;
        has_sent_first_goal_ = true;
        last_goal_published_time_ = current_time; 
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    message_filters::Subscriber<geometry_msgs::msg::PolygonStamped> bbox_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<geometry_msgs::msg::PolygonStamped, sensor_msgs::msg::Image>;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    double fx_, fy_, cx_, cy_;
    bool has_calibration_;
    rclcpp::Time last_goal_published_time_;
    geometry_msgs::msg::Point last_sent_odom_point_;
    bool has_sent_first_goal_ = false;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    std::shared_ptr<TargetProjectorNode> node = std::make_shared<TargetProjectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

