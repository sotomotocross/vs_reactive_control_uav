#pragma once

#include <ros/ros.h>

#include "vs_reactive_control_controller/Controller.hpp"
#include "vs_reactive_control_controller/FeatureData.hpp"

#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#include "img_seg_cnn/PredData.h"
#include "img_seg_cnn/PolyCalcCustom.h"
#include "img_seg_cnn/PolyCalcCustomTF.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace vs_reactive_control_controller
{
  // UtilityFunctions class contains various utility functions used in the controller
  class UtilityFunctions
  {
  public:
    
    // Function for transforming camera velocities to UAV velocities
    static Eigen::MatrixXd VelTrans(Eigen::MatrixXd CameraVel);
    // Another version of VelTrans function with a different parameter
    static Eigen::MatrixXd VelTrans1(Eigen::MatrixXd CameraVel1);
    
    // Function for calculating dynamics based on feature properties and indices
    Eigen::MatrixXd Dynamics(Eigen::VectorXd feat_prop,
                                               int transformed_first_min_index, 
                                               int transformed_second_min_index,
                                               double transformed_s_bar_x, double transformed_s_bar_y,
                                               double transformed_tangent, double Z0);

    // Functions for calculating gradient of basis functions with respect to x1, x2, x3, and x4
    static Eigen::MatrixXd grad_basis_x1(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x2(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x3(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x4(Eigen::VectorXd x);

    // Function for loading weights from a file
    static Eigen::MatrixXd weights_loading(std::string filename);

  private:
  };
} // namespace vs_reactive_control_controller