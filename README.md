# Vision-based Reactive control on UAVs for target tracking
This is a ROS C++ package of vision-based Reactive Control strategy for UAV aiming on target tracking. The general application of thiese controllers is a dynamic coastline in the presenc of waves. The different version of visual servoing will be analyzed belowThe controllers were all developed in a UAV synthetic simulation environment: https://github.com/sotomotocross/UAV_simulator_ArduCopter.git

# Things to Do 
1) Check the results based on the recorded rosbags from the simulation sessions and make plots in order to present the results. Is there something to change for now in the framework?
2) Is there a point of implementing a ROSservice in order to synchronize the perception node (Controller::featureCallback_poly_custom_tf) with the controller node running in the current ros package?
3) Multiple simulations and implementation on the NVIDIA AGX Xavier Developer Kit and testing?
4) Implementation of RBFs on the vs_reactive_control_rbf_controller???
5) 