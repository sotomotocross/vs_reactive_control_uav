#include "vs_reactive_control_uav_target_tracking/controller.hpp"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vs_reactive_control_uav_target_tracking");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");


  std::cout << "starting vs_reactive_control_uav_target_tracking_node" << std::endl;

  vs_reactive_control_uav_target_tracking::Controller controller(nh, pnh);

  ros::spin();

  return 0;
}
