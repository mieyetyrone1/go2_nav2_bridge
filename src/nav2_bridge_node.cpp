#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

class Go2Nav2Bridge : public rclcpp::Node
{
public:
    Go2Nav2Bridge() : Node("go2_nav2_bridge"), sport_client_(this)
    {
        // 1. Core control publisher
        sport_req_pub_ = this->create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", 10);

        // 2. Subscriber for incoming Nav2 geometry twist vectors
        twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&Go2Nav2Bridge::twistCallback, this, std::placeholders::_1));

        boot_time_ = this->get_clock()->now();
        msg_counter_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Go2 Nav2 Explicit Handshake Bridge Ready!");
    }

private:
    void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        auto now = this->get_clock()->now();
        
        // Safety Gate: Let network paths settle for the first 1.5 seconds
        if ((now - boot_time_).seconds() < 1.5) {
            return;
        }

        float vx = msg->linear.x;
        float vy = msg->linear.y;
        float vyaw = msg->angular.z;

        // CRITICAL FIX: Increment the identity transaction ID tracker. 
        // Unitree's controller drops messages if the ID remains statically 0.
        msg_counter_++;
        req_.header.identity.id = msg_counter_;
        req_.header.policy.priority = 0;

        // Parse speeds into our signed request package
        sport_client_.Move(req_, vx, vy, vyaw);

        // Broadcast directly to the high-level motion controller channel
        sport_req_pub_->publish(req_);
    }

    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_req_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    
    SportClient sport_client_;       
    unitree_api::msg::Request req_;  
    rclcpp::Time boot_time_;
    int64_t msg_counter_; // Handshake transaction tracker
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


