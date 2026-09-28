#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

class Go2Nav2Bridge : public rclcpp::Node
{
public:
    Go2Nav2Bridge() : Node("go2_nav2_bridge"), sport_client_(this)
    {
        sport_req_pub_ = this->create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", 10);

        twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&Go2Nav2Bridge::twistCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/utlidar/robot_odom", 10, std::bind(&Go2Nav2Bridge::odomCallback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        boot_time_ = this->get_clock()->now();
        msg_counter_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Go2 Nav2 Unified Clock Bridge Ready!");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        auto hardware_time = msg->header.stamp;

        // 1. DYNAMIC TRANSFORM: odom -> base_link
        geometry_msgs::msg::TransformStamped odom_tf;
        odom_tf.header.stamp = hardware_time;
        odom_tf.header.frame_id = "odom";
        odom_tf.child_frame_id = "base_link";
        odom_tf.transform.translation.x = msg->pose.pose.position.x;
        odom_tf.transform.translation.y = msg->pose.pose.position.y;
        odom_tf.transform.translation.z = msg->pose.pose.position.z;
        odom_tf.transform.rotation = msg->pose.pose.orientation;
        tf_broadcaster_->sendTransform(odom_tf);

        // 2. HARDWARE-LOCKED STATIC ANCHOR: map -> odom
        geometry_msgs::msg::TransformStamped map_tf;
        map_tf.header.stamp = hardware_time;
        map_tf.header.frame_id = "map";
        map_tf.child_frame_id = "odom";
        map_tf.transform.translation.x = 0.0;
        map_tf.transform.translation.y = 0.0;
        map_tf.transform.translation.z = 0.0;
        map_tf.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(map_tf);

        // 3. HARDWARE-LOCKED LIDAR TRANSFORM: base_link -> utlidar_lidar
        geometry_msgs::msg::TransformStamped lidar_tf;
        lidar_tf.header.stamp = hardware_time;
        lidar_tf.header.frame_id = "base_link";
        lidar_tf.child_frame_id = "utlidar_lidar";
        lidar_tf.transform.translation.x = 0.25;
        lidar_tf.transform.translation.y = 0.0;
        lidar_tf.transform.translation.z = 0.15;
        lidar_tf.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(lidar_tf);
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

        msg_counter_++;
        req_.header.identity.id = msg_counter_;
        req_.header.policy.priority = 0;

        sport_client_.Move(req_, vx, vy, vyaw);
        sport_req_pub_->publish(req_);
    }

    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_req_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    SportClient sport_client_;       
    unitree_api::msg::Request req_;  
    rclcpp::Time boot_time_;
    int64_t msg_counter_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Go2Nav2Bridge>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

