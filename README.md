# Vision-based Reactive control on UAVs for target tracking
This is a ROS C++ package of vision-based Reactive Control strategy for UAV aiming on target tracking. The general application of thiese controllers is a dynamic coastline in the presenc of waves. The different version of visual servoing will be analyzed belowThe controllers were all developed in a UAV synthetic simulation environment: https://github.com/sotomotocross/UAV_simulator_ArduCopter.git

# vs_reactive_control_uav

This repository contains a set of ROS packages for implementing reactive control for UAVs (Unmanned Aerial Vehicles) using computer vision techniques.

## Packages

### 1. vs_reactive_control_controller

#### Description

The `vs_reactive_control_controller` package implements the reactive control logic for UAVs. It utilizes computer vision data and features for control decision-making.

#### Dependencies

- ROS (Robot Operating System)
- Eigen library
- mavros package
- img_seg_cnn package

#### Build Instructions

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/vs_reactive_control_uav.git
    ```

2. Build the catkin workspace:

    ```bash
    cd vs_reactive_control_uav
    catkin_make
    ```

#### Running the Controller

After building the workspace, you can run the controller node:

```bash
rosrun vs_reactive_control_controller vs_reactive_control_controller_node
```

### 2. vs_reactive_control_rbf_controller (To be implemented)
