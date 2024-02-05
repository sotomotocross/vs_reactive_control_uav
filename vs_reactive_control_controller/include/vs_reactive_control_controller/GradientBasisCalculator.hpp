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
  class GradientBasisCalculator
  {
  public:
    static Eigen::MatrixXd grad_basis_x1(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x2(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x3(Eigen::VectorXd x);
    static Eigen::MatrixXd grad_basis_x4(Eigen::VectorXd x);
    // Add any other gradient basis functions here
  private:
    // Add any private members or helper functions for gradient basis calculations
  };
} // namespace vs_reactive_control_controller
