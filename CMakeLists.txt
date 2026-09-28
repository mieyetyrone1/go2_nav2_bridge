cmake_minimum_required(VERSION 3.8)
project(go2_nav2_bridge)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

# Find all required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# TARGET 1: Original Nav2 Hardware Bridge
add_executable(nav2_bridge_node
  src/nav2_bridge_node.cpp
  ${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/src/common/ros2_sport_client.cpp
)

target_include_directories(nav2_bridge_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/include/common>
)

ament_target_dependencies(nav2_bridge_node 
  rclcpp 
  geometry_msgs 
  unitree_go 
  unitree_api 
  nav_msgs 
  tf2_ros
)

# TARGET 2: New Target Projection Node
add_executable(target_projector_node
  src/target_projector_node.cpp
)

ament_target_dependencies(target_projector_node 
  rclcpp 
  sensor_msgs 
  geometry_msgs 
  message_filters 
  tf2_ros 
  tf2_geometry_msgs
)

# Installation Rules for Executables & Assets
install(TARGETS
  nav2_bridge_node
  target_projector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

