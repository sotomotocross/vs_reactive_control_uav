
cmake_minimum_required(VERSION 3.0.2)
project(vs_reactive_control_controller)

# Set C++ standard and flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-g")

# Find catkin dependencies
find_package(catkin REQUIRED
    COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    message_generation
    mavros
    mavros_msgs
    img_seg_cnn
    sensor_msgs
)

# Declare the package dependencies
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp std_msgs geometry_msgs message_runtime mavros mavros_msgs img_seg_cnn sensor_msgs
    DEPENDS system_lib eigen3
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vs_reactive_control_controller
)

# Declare the library sources
set(LIB_SOURCES
    # src/controller.cpp
    src/Controller.cpp
    src/FeatureData.cpp
    src/VelocityTransformer.cpp
    src/DynamicsCalculator.cpp
    src/GradientBasisCalculator.cpp
    src/WeightLoader.cpp
    # src/UtilityFunctions.cpp
)

# Declare a library
add_library(${PROJECT_NAME}
    ${LIB_SOURCES}
)

# Add library dependencies
add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

# Link the library
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

# Declare an executable
add_executable(${PROJECT_NAME}_node
    src/node.cpp
)

# Add executable dependencies
add_dependencies(${PROJECT_NAME}_node
    ${catkin_EXPORTED_TARGETS}
)

# Link the executable with the library
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)
