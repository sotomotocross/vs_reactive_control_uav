#include "vs_reactive_control_rbf_controller/controller.hpp"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vs_reactive_control_rbf_controller");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::cout << "starting vs_reactive_control_rbf_controller_node" << std::endl;

  vs_reactive_control_rbf_controller::Controller controller(nh, pnh);

  ros::spin();

  return 0;
}
