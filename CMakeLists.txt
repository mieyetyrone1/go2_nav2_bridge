cmake_minimum_required(VERSION 3.5)
project(go2_nav2_bridge)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14. Changed to 17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Add nav2_bridge_node
add_executable(nav2_bridge_node 
  src/nav2_bridge_node.cpp
  ${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/src/common/ros2_sport_client.cpp
)

# Direct CMake to pull inclusion hooks from Unitree's custom example include directories
target_include_directories(nav2_bridge_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../unitree_ros2/example/src/include/common> # === ADD THIS LINE ===
)


ament_target_dependencies(nav2_bridge_node rclcpp geometry_msgs unitree_api nav_msgs tf2_ros)

install(TARGETS
  nav2_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the launch and config directories into the global workspace share path
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
