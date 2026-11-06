#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "action_msgs/srv/cancel_goal.hpp"

class Go2Nav2Bridge : public rclcpp::Node
{
public:
    Go2Nav2Bridge() : Node("go2_nav2_bridge"), sport_client_(this)
    {
        sport_req_pub_ = this->create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", 10);

        twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&Go2Nav2Bridge::twistCallback, this, std::placeholders::_1));

        auto hardware_qos = rclcpp::QoS(rclcpp::KeepLast(10));
        hardware_qos.reliable();                            
        hardware_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE); 

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/utlidar/robot_odom", hardware_qos, std::bind(&Go2Nav2Bridge::odomCallback, this, std::placeholders::_1));

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/utlidar/cloud_deskewed", hardware_qos, std::bind(&Go2Nav2Bridge::pointcloudCallback, this, std::placeholders::_1));

        nav_feedback_sub_ = this->create_subscription<nav2_msgs::action::NavigateToPose::Impl::FeedbackMessage>(
            "/navigate_to_pose/_action/feedback", 10, std::bind(&Go2Nav2Bridge::feedbackCallback, this, std::placeholders::_1));

        nav_cancel_client_ = this->create_client<action_msgs::srv::CancelGoal>("/navigate_to_pose/_action/cancel_goal");

        // FIXED TYPO: Added missing class member variable underscore suffix
        scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", rclcpp::QoS(10).reliable());

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        boot_time_ = this->get_clock()->now();
        msg_counter_ = 0;
        odom_dbg_counter_ = 0;
        cloud_dbg_counter_ = 0;
        
        time_offset_calculated_ = false;
        time_offset_seconds_ = 0.0;
        distance_remaining_ = 999.0;

        RCLCPP_INFO(this->get_logger(), "Go2 Nav2 Velocity Diagnostic Bridge Active!");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        auto live_jetson_time = this->get_clock()->now();

        if (!time_offset_calculated_) {
            double hw_time = msg->header.stamp.sec + (msg->header.stamp.nanosec * 1e-9);
            double jetson_time = live_jetson_time.seconds();
            time_offset_seconds_ = jetson_time - hw_time;
            time_offset_calculated_ = true;
            std::cout << "\n=======================================================" << std::endl;
            std::cout << "[SUCCESS] Automated Time Mapping Engine Calibrated!" << std::endl;
            std::cout << "[SUCCESS] Calculated Clock Offset: " << time_offset_seconds_ << " seconds" << std::endl;
            std::cout << "=======================================================\n" << std::endl;
        }

        odom_dbg_counter_++;
        auto corrected_stamp = live_jetson_time;

        geometry_msgs::msg::TransformStamped odom_tf;
        odom_tf.header.stamp = corrected_stamp;
        odom_tf.header.frame_id = "odom";
        odom_tf.child_frame_id = "base_link";
        odom_tf.transform.translation.x = msg->pose.pose.position.x;
        odom_tf.transform.translation.y = msg->pose.pose.position.y;
        odom_tf.transform.translation.z = msg->pose.pose.position.z;
        odom_tf.transform.rotation = msg->pose.pose.orientation;
        tf_broadcaster_->sendTransform(odom_tf);
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!time_offset_calculated_) return;

        cloud_dbg_counter_++;
        auto live_jetson_time = this->get_clock()->now();
        auto scan_msg = std::make_shared<sensor_msgs::msg::LaserScan>();
        scan_msg->header.stamp = live_jetson_time; 
        scan_msg->header.frame_id = "utlidar_lidar";
        scan_msg->angle_min = -3.14159f;
        scan_msg->angle_max = 3.14159f;
        scan_msg->angle_increment = 0.0087f;
        scan_msg->time_increment = 0.0f;
        scan_msg->scan_time = 0.1f;
        scan_msg->range_min = 0.2f;
        scan_msg->range_max = 20.0f;

        int num_readings = std::ceil((scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);
        scan_msg->ranges.assign(num_readings, std::numeric_limits<float>::infinity());

        int x_offset = -1, y_offset = -1, z_offset = -1;
        for (const auto& field : msg->fields) {
            if (field.name == "x") x_offset = field.offset;
            if (field.name == "y") y_offset = field.offset;
            if (field.name == "z") z_offset = field.offset;
        }

        if (x_offset != -1 && y_offset != -1 && z_offset != -1) {
            for (size_t i = 0; i < msg->data.size(); i += msg->point_step) {
                float x = *reinterpret_cast<const float*>(&msg->data[i + x_offset]);
                float y = *reinterpret_cast<const float*>(&msg->data[i + y_offset]);
                float z = *reinterpret_cast<const float*>(&msg->data[i + z_offset]);

                if (z >= -0.3f && z <= 0.5f) {
                    float range = std::hypot(x, y);
                    if (range >= scan_msg->range_min && range <= scan_msg->range_max) {
                        float angle = std::atan2(y, x);
                        int index = std::floor((angle - scan_msg->angle_min) / scan_msg->angle_increment);
                        if (index >= 0 && index < num_readings) {
                            if (range < scan_msg->ranges[index]) {
                                scan_msg->ranges[index] = range;
                            }
                        }
                    }
                }
            }
        }
        scan_pub_->publish(*scan_msg);
    }

    void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        auto now = this->get_clock()->now();
        if ((now - boot_time_).seconds() < 1.5) {
            return;
        }

        float vx = msg->linear.x;
        float vy = msg->linear.y;
        float vyaw = msg->angular.z;
        
        std::cout << "[TWIST RECEIVED] vx: " << vx << " | vy: " << vy << " | vyaw: " << vyaw << " | Dist Remaining: " << distance_remaining_ << std::endl;
        
        // Trigger early stop if we are 0.7 meters (relative to base_link) or closer to the destination
        if (distance_remaining_ <= 0.70) {
            std::cout << "[EARLY STOP TRIGGERED] Robot is " << distance_remaining_ << "m away. Stopping and canceling goal." << std::endl;
            // Force all commands to absolute zero instantly
            vx = 0.0;
            vy = 0.0;
            vyaw = 0.0;
            // Tell Nav2 to cancel the active goal so the action server finishes cleanly
            if (nav_cancel_client_->service_is_ready()) {
                auto cancel_req = std::make_shared<action_msgs::srv::CancelGoal::Request>();
                nav_cancel_client_->async_send_request(cancel_req);
                std::cout << "[CANCEL SENT] Sent clean cancel to Nav2 action server." << std::endl;
            }
            // Reset distance tracking so it doesn't repeatedly spam cancel requests
            distance_remaining_ = 999.0;
        }

        msg_counter_++;
        req_.header.identity.id = msg_counter_;
        req_.header.policy.priority = 0;
        
        sport_client_.Move(req_, vx, vy, vyaw);
        std::cout << "[SPORT CLIENT] Forwarding Request Token ID: " << msg_counter_ << std::endl;
        sport_req_pub_->publish(req_);
    }

    void feedbackCallback(const nav2_msgs::action::NavigateToPose::Impl::FeedbackMessage::SharedPtr msg)
    {
        distance_remaining_ = msg->feedback.distance_remaining;
    }

    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_req_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav2_msgs::action::NavigateToPose::Impl::FeedbackMessage>::SharedPtr nav_feedback_sub_;
    rclcpp::Client<action_msgs::srv::CancelGoal>::SharedPtr nav_cancel_client_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    SportClient sport_client_;
    unitree_api::msg::Request req_;
    rclcpp::Time boot_time_;
    float distance_remaining_;	
    int64_t msg_counter_;
    int64_t odom_dbg_counter_;
    int64_t cloud_dbg_counter_;
    bool time_offset_calculated_;
    double time_offset_seconds_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Go2Nav2Bridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

