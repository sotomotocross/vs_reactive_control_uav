cmake_minimum_required(VERSION 3.0.2)
project(vs_reactive_control_controller)

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "-g")

set(PACKAGE_DEPENDENCIES
    roscpp
    std_msgs
    geometry_msgs
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
    geometry_msgs
    message_generation
    roscpp
    rospy
    std_msgs
    img_seg_cnn
    mavros
    mavros_msgs
    sensor_msgs
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS ${PACKAGE_DEPENDS} 
    message_runtime 
    geometry_msgs 
    message_generation 
    roscpp 
    rospy 
    std_msgs 
    img_seg_cnn 
    mavros 
    mavros_msgs 
    sensor_msgs
    DEPENDS system_lib eigen3
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
    src/controller.cpp
)

add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
    src/node.cpp
)

add_dependencies(${PROJECT_NAME}_node
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)
