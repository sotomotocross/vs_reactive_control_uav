#pragma once

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
  class DynamicsCalculator
  {
  public:
    Eigen::MatrixXd Dynamics(Eigen::VectorXd feat_prop,
                             int transformed_first_min_index,
                             int transformed_second_min_index,
                             double transformed_s_bar_x, double transformed_s_bar_y,
                             double transformed_tangent, double Z0);
    // Add any other dynamics-related functions here
  private:
    // Add any private members or helper functions for dynamics calculations
  };
} // namespace vs_reactive_control_controller
