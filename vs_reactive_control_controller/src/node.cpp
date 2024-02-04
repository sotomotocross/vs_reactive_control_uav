#include "vs_reactive_control_controller/Controller.hpp"

int main(int argc, char* argv[])
{
  // Initialize ROS
  ros::init(argc, argv, "vs_reactive_control_controller");

  // Create a ROS node handle
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Print a message indicating the start of the node
  std::cout << "Starting vs_reactive_control_controller_node" << std::endl;

  // Create an instance of the Controller class
  vs_reactive_control_controller::Controller controller(nh, pnh);

  // Enter the ROS spin loop
  ros::spin();

  return 0;
}
