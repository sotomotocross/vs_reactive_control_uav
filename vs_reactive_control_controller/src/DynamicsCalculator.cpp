#include "vs_reactive_control_controller/DynamicsCalculator.hpp"

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
  Eigen::MatrixXd DynamicsCalculator::Dynamics(Eigen::VectorXd feat_prop,
                                             int first, int second,
                                             double transformed_s_bar_x, 
                                             double transformed_s_bar_y,
                                             double transformed_tangent, double Z0)
  {
    int dim_s = 4;
    int dim_inputs = 4;
    MatrixXd model_mat(dim_s, dim_inputs);

    // Barycenter dynamics calculation
    double term_1_4 = 0.0;
    double term_1_5 = 0.0;
    double term_2_4 = 0.0;
    double term_2_5 = 0.0;

    int N;
    N = feat_prop.size() / 2;
    // cout << "\nN = " << N << endl;
    // cout << "first = " << first << endl;
    // cout << "second = " << second << endl;
    // cout << "transformed_first_min_index = " << transformed_first_min_index << endl;
    // cout << "transformed_second_min_index = " << transformed_second_min_index << endl;
    // cout << "first_min_index = " << first_min_index << endl;
    // cout << "second_min_index = " << second_min_index << endl;

    // if ((first == 0 && second == N - 1) || (first == N - 1 && second == 0))
    // {

    for (int i = 0; i < N - 1; i += 2)
    {
      term_1_4 = term_1_4 + feat_prop[i] * feat_prop[i + 1];
      term_1_5 = term_1_5 + (1 + pow(feat_prop[i], 2));
      term_2_4 = term_2_4 + (1 + pow(feat_prop[i + 1], 2));
      term_2_5 = term_2_5 + feat_prop[i] * feat_prop[i + 1];
    }

    term_1_4 = term_1_4 / N;
    term_1_5 = -term_1_5 / N;
    term_2_4 = term_2_4 / N;
    term_2_5 = -term_2_5 / N;

    double g_4_4, g_4_5, g_4_6;

    // Angle dynamics calculation
    // Fourth term
    double term_4_4_1, term_4_4_2, term_4_4_3, term_4_4_4;
    double sum_4_4_1 = 0.0, sum_4_4_2 = 0.0;

    double k = 0;
    VectorXd x(N);
    VectorXd y(N);

    for (int i = 0; i < 2 * N - 1; i += 2)
    {
      x[k] = feat_prop[i];
      k++;
    }

    k = 0;

    for (int i = 1; i < 2 * N; i += 2)
    {
      y[k] = feat_prop[i];
      k++;
    }

    for (int i = 0; i < N - 1; i += 2)
    {
      // cout << "i = " << i << endl;
      sum_4_4_1 = sum_4_4_1 + pow(feat_prop[i + 1], 2);
      sum_4_4_2 = sum_4_4_2 + feat_prop[i] * feat_prop[i + 1];
    }

    term_4_4_1 = transformed_tangent / (y[first] + y[second] - 2 * transformed_s_bar_y);
    term_4_4_2 = (pow(y[first], 2) + pow(y[second], 2) - (2 / N) * sum_4_4_1);
    term_4_4_3 = -1 / (y[first] + y[second] - 2 * transformed_s_bar_y);
    term_4_4_4 = (x[first] * y[first] + x[second] * y[second] - (2 / N) * sum_4_4_2);

    g_4_4 = term_4_4_1 * term_4_4_2 + term_4_4_3 * term_4_4_4;

    // Fifth term
    double term_4_5_1, term_4_5_2, term_4_5_3, term_4_5_4;
    double sum_4_5_1 = 0.0, sum_4_5_2 = 0.0;

    for (int i = 0; i < N - 1; i += 2)
    {
      // cout << "i = " << i << endl;
      sum_4_5_1 = sum_4_5_1 + pow(feat_prop[i], 2);
      sum_4_5_2 = sum_4_5_2 + feat_prop[i] * feat_prop[i + 1];
    }

    term_4_5_1 = 1 / (y[first] + y[second] - 2 * transformed_s_bar_y);
    term_4_5_2 = (pow(x[first], 2) + pow(x[second], 2) - (2 / N) * sum_4_5_1);
    term_4_5_3 = -transformed_tangent / (y[first] + y[second] - 2 * transformed_s_bar_y);
    term_4_5_4 = (x[first] * y[first] + x[second] * y[second] - (2 / N) * sum_4_5_2);

    g_4_5 = term_4_5_1 * term_4_5_2 + term_4_5_3 * term_4_5_4;

    // Fifth term
    g_4_6 = pow(transformed_tangent, 2) + 1;

    model_mat << -1 / Z0, 0.0, transformed_s_bar_x / Z0, transformed_s_bar_y,
        0.0, -1 / Z0, transformed_s_bar_y / Z0, -transformed_s_bar_x,
        0.0, 0.0, 2 / Z0, 0.0,
        0.0, 0.0, 0.0, g_4_6;
    // }

    // cout << "model calculation before returning to main!" << endl;
    // cout << "model_mat = \n" << model_mat << endl;

    return model_mat;
  }
} // namespace vs_reactive_control_controller
