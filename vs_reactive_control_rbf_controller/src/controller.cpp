#include "vs_reactive_control_rbf_controller/controller.hpp"

#include <thread>
#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

namespace vs_reactive_control_rbf_controller
{

  Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
  {
    // variables initialization
    cX = 0.0;
    cY = 0.0;
    cX_int = 0;
    cY_int = 0;
    Z0 = 0.0;
    Z1 = 0.0;
    Z2 = 0.0;
    Z3 = 0.0;
    s_bar_x = 0.0;
    s_bar_y = 0.0;
    custom_sigma = 1.0;
    custom_sigma_square = 1.0;
    custom_sigma_square_log = 1.0;
    angle_tangent = 0.0;
    angle_radian = 0.0;
    angle_deg = 0.0;
    transformed_s_bar_x = 0.0;
    transformed_s_bar_y = 0.0;
    transformed_sigma = 1.0;
    transformed_sigma_square = 1.0;
    transformed_sigma_square_log = 1.0;
    transformed_tangent = 0.0;
    transformed_angle_radian = 0.0;
    transformed_angle_deg = 0.0;

    pnh_.param<double>("reactive_controller/forward_term", forward_term, 1.0);
    pnh_.param<double>("reactive_controller/gain_tx", gain_tx, 1.0);
    pnh_.param<double>("reactive_controller/gain_ty", gain_ty, 1.0);
    pnh_.param<double>("reactive_controller/gain_tz", gain_tz, 2.0);
    pnh_.param<double>("reactive_controller/gain_yaw", gain_yaw, 1.0);

    // ROS subscribers
    feature_sub_poly_custom_ = nh_.subscribe("polycalc_custom", 10, &Controller::featureCallback_poly_custom, this);
    feature_sub_poly_custom_tf_ = nh.subscribe("polycalc_custom_tf", 10, &Controller::featureCallback_poly_custom_tf, this);
    alt_sub_ = nh.subscribe("/mavros/global_position/rel_alt", 10, &Controller::altitudeCallback, this);

    // ROS publishers
    vel_pub_ = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    state_vec_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
    state_vec_des_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);

    std::thread control_loop_thread(&Controller::update, this);
    control_loop_thread.detach();
  }

  Controller::~Controller()
  {
    vel_pub_.shutdown();
  }

  MatrixXd Controller::Dynamics(VectorXd feat_prop, int first, int second)
  {
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

  MatrixXd Controller::grad_basis_x1(VectorXd x)
  {
    MatrixXd B_x1;
    B_x1.setZero(1, 75);

    B_x1(0, 0) = 0.0;
    B_x1(0, 1) = 1.0;
    B_x1(0, 2) = 0.0;
    B_x1(0, 3) = 0.0;
    B_x1(0, 4) = 0.0;
    B_x1(0, 5) = x(1);
    B_x1(0, 6) = x(2);
    B_x1(0, 7) = x(3);
    B_x1(0, 8) = 0.0;
    B_x1(0, 9) = 0.0;
    B_x1(0, 10) = 0.0;
    B_x1(0, 11) = 2 * x(0);
    B_x1(0, 12) = 0.0;
    B_x1(0, 13) = 0.0;
    B_x1(0, 14) = 0.0;
    B_x1(0, 15) = 2 * x(0) * x(1);
    B_x1(0, 16) = 2 * x(0) * x(2);
    B_x1(0, 17) = 2 * x(0) * x(3);
    B_x1(0, 18) = pow(x(1), 2);
    B_x1(0, 19) = 0.0;
    B_x1(0, 20) = 0.0;
    B_x1(0, 21) = pow(x(2), 2);
    B_x1(0, 22) = 0.0;
    B_x1(0, 23) = 0.0;
    B_x1(0, 24) = pow(x(3), 2);
    B_x1(0, 25) = 0.0;
    B_x1(0, 26) = 0.0;
    B_x1(0, 27) = 3 * pow(x(0), 2);
    B_x1(0, 28) = 0.0;
    B_x1(0, 29) = 0.0;
    B_x1(0, 30) = 0.0;
    B_x1(0, 31) = 3 * pow(x(0), 2) * x(1);
    B_x1(0, 32) = 3 * pow(x(0), 2) * x(2);
    B_x1(0, 33) = 3 * pow(x(0), 2) * x(3);
    B_x1(0, 34) = pow(x(1), 3);
    B_x1(0, 35) = 0.0;
    B_x1(0, 36) = 0.0;
    B_x1(0, 37) = pow(x(2), 3);
    B_x1(0, 38) = 0.0;
    B_x1(0, 39) = 0.0;
    B_x1(0, 40) = pow(x(3), 3);
    B_x1(0, 41) = 0.0;
    B_x1(0, 42) = 0.0;
    B_x1(0, 43) = 3 * pow(x(0), 2) * pow(x(1), 2);
    B_x1(0, 44) = 3 * pow(x(0), 2) * pow(x(2), 2);
    B_x1(0, 45) = 3 * pow(x(0), 2) * pow(x(3), 2);
    B_x1(0, 46) = 2 * pow(x(1), 3) * x(0);
    B_x1(0, 47) = 0.0;
    B_x1(0, 48) = 0.0;
    B_x1(0, 49) = 2 * pow(x(2), 3) * x(0);
    B_x1(0, 50) = 0.0;
    B_x1(0, 51) = 0.0;
    B_x1(0, 52) = 2 * pow(x(3), 3) * x(0);
    B_x1(0, 53) = 0.0;
    B_x1(0, 54) = 0.0;
    B_x1(0, 55) = 4 * pow(x(0), 3);
    B_x1(0, 56) = 0.0;
    B_x1(0, 57) = 0.0;
    B_x1(0, 58) = 0.0;
    B_x1(0, 59) = 4 * pow(x(0), 3) * x(1);
    B_x1(0, 60) = 4 * pow(x(0), 3) * x(2);
    B_x1(0, 61) = 4 * pow(x(0), 3) * x(3);
    B_x1(0, 62) = pow(x(1), 4);
    B_x1(0, 63) = 0.0;
    B_x1(0, 64) = 0.0;
    B_x1(0, 65) = pow(x(2), 4);
    B_x1(0, 66) = 0.0;
    B_x1(0, 67) = 0.0;
    B_x1(0, 68) = pow(x(3), 4);
    B_x1(0, 69) = 0.0;
    B_x1(0, 70) = 0.0;
    B_x1(0, 71) = 5 * pow(x(0), 4);
    B_x1(0, 72) = 0.0;
    B_x1(0, 73) = 0.0;
    B_x1(0, 74) = 0.0;

    return B_x1;
  }

  MatrixXd Controller::grad_basis_x2(VectorXd x)
  {
    MatrixXd B_x2;
    B_x2.setZero(1, 75);

    B_x2(0, 0) = 0.0;
    B_x2(0, 1) = 0.0;
    B_x2(0, 2) = 1.0;
    B_x2(0, 3) = 0.0;
    B_x2(0, 4) = 0.0;
    B_x2(0, 5) = x(0);
    B_x2(0, 6) = 0.0;
    B_x2(0, 7) = 0.0;
    B_x2(0, 8) = x(2);
    B_x2(0, 9) = x(3);
    B_x2(0, 10) = 0.0;
    B_x2(0, 11) = 0.0;
    B_x2(0, 12) = 2 * x(1);
    B_x2(0, 13) = 0.0;
    B_x2(0, 14) = 0.0;
    B_x2(0, 15) = pow(x(0), 2);
    B_x2(0, 16) = 0.0;
    B_x2(0, 17) = 0.0;
    B_x2(0, 18) = 2 * x(1) * x(0);
    B_x2(0, 19) = 2 * x(1) * x(2);
    B_x2(0, 20) = 2 * x(1) * x(3);
    B_x2(0, 21) = 0.0;
    B_x2(0, 22) = pow(x(2), 2);
    B_x2(0, 23) = 0.0;
    B_x2(0, 24) = 0.0;
    B_x2(0, 25) = pow(x(3), 2);
    B_x2(0, 26) = 0.0;
    B_x2(0, 27) = 0.0;
    B_x2(0, 28) = 3 * pow(x(1), 2);
    B_x2(0, 29) = 0.0;
    B_x2(0, 30) = 0.0;
    B_x2(0, 31) = pow(x(0), 3);
    B_x2(0, 32) = 0.0;
    B_x2(0, 33) = 0.0;
    B_x2(0, 34) = 3 * pow(x(1), 2) * x(0);
    B_x2(0, 35) = 3 * pow(x(1), 2) * x(2);
    B_x2(0, 36) = 3 * pow(x(1), 2) * x(3);
    B_x2(0, 37) = 0.0;
    B_x2(0, 38) = pow(x(2), 3);
    B_x2(0, 39) = 0.0;
    B_x2(0, 40) = 0.0;
    B_x2(0, 41) = pow(x(3), 3);
    B_x2(0, 42) = 0.0;
    B_x2(0, 43) = 2 * pow(x(0), 3) * x(1);
    B_x2(0, 44) = 0.0;
    B_x2(0, 45) = 0.0;
    B_x2(0, 46) = 3 * pow(x(1), 2) * pow(x(0), 2);
    B_x2(0, 47) = 3 * pow(x(1), 2) * pow(x(2), 2);
    B_x2(0, 48) = 3 * pow(x(1), 2) * pow(x(3), 2);
    B_x2(0, 49) = 0.0;
    B_x2(0, 50) = 2 * pow(x(2), 3) * x(1);
    B_x2(0, 51) = 0.0;
    B_x2(0, 52) = 0.0;
    B_x2(0, 53) = 2 * pow(x(3), 3) * x(1);
    B_x2(0, 54) = 0.0;
    B_x2(0, 55) = 0.0;
    B_x2(0, 56) = 4 * pow(x(1), 3);
    B_x2(0, 57) = 0.0;
    B_x2(0, 58) = 0.0;
    B_x2(0, 59) = pow(x(0), 4);
    B_x2(0, 60) = 0.0;
    B_x2(0, 61) = 0.0;
    B_x2(0, 62) = 4 * pow(x(1), 3) * x(0);
    B_x2(0, 63) = 4 * pow(x(1), 3) * x(2);
    B_x2(0, 64) = 4 * pow(x(1), 3) * x(3);
    B_x2(0, 65) = 0.0;
    B_x2(0, 66) = pow(x(2), 4);
    B_x2(0, 67) = 0.0;
    B_x2(0, 68) = 0.0;
    B_x2(0, 69) = pow(x(3), 4);
    B_x2(0, 70) = 0.0;
    B_x2(0, 71) = 0.0;
    B_x2(0, 72) = 5 * pow(x(1), 4);
    B_x2(0, 73) = 0.0;
    B_x2(0, 74) = 0.0;

    return B_x2;
  }

  MatrixXd Controller::grad_basis_x3(VectorXd x)
  {
    MatrixXd B_x3;
    B_x3.setZero(1, 75);

    B_x3(0, 0) = 0.0;
    B_x3(0, 1) = 0.0;
    B_x3(0, 2) = 0.0;
    B_x3(0, 3) = 1.0;
    B_x3(0, 4) = 0.0;
    B_x3(0, 5) = 0.0;
    B_x3(0, 6) = x(0);
    B_x3(0, 7) = 0.0;
    B_x3(0, 8) = x(1);
    B_x3(0, 9) = 0.0;
    B_x3(0, 10) = x(3);
    B_x3(0, 11) = 0.0;
    B_x3(0, 12) = 0.0;
    B_x3(0, 13) = 2 * x(2);
    B_x3(0, 14) = 0.0;
    B_x3(0, 15) = 0.0;
    B_x3(0, 16) = pow(x(0), 2);
    B_x3(0, 17) = 0.0;
    B_x3(0, 18) = 0.0;
    B_x3(0, 19) = pow(x(1), 2);
    B_x3(0, 20) = 0.0;
    B_x3(0, 21) = 2 * x(2) * x(0);
    B_x3(0, 22) = 2 * x(2) * x(1);
    B_x3(0, 23) = 2 * x(2) * x(3);
    B_x3(0, 24) = 0.0;
    B_x3(0, 25) = 0.0;
    B_x3(0, 26) = pow(x(3), 2);
    B_x3(0, 27) = 0.0;
    B_x3(0, 28) = 0.0;
    B_x3(0, 29) = 3 * pow(x(2), 2);
    B_x3(0, 30) = 0.0;
    B_x3(0, 31) = 0.0;
    B_x3(0, 32) = pow(x(0), 3);
    B_x3(0, 33) = 0.0;
    B_x3(0, 34) = 0.0;
    B_x3(0, 35) = pow(x(1), 3);
    B_x3(0, 36) = 0.0;
    B_x3(0, 37) = 3 * pow(x(2), 2) * x(0);
    B_x3(0, 38) = 3 * pow(x(2), 2) * x(1);
    B_x3(0, 39) = 3 * pow(x(2), 2) * x(3);
    B_x3(0, 40) = 0.0;
    B_x3(0, 41) = 0.0;
    B_x3(0, 42) = pow(x(3), 3);
    B_x3(0, 43) = 0.0;
    B_x3(0, 44) = 2 * pow(x(0), 3) * x(2);
    B_x3(0, 45) = 0.0;
    B_x3(0, 46) = 0.0;
    B_x3(0, 47) = 2 * pow(x(1), 3) * x(2);
    B_x3(0, 48) = 0.0;
    B_x3(0, 49) = 3 * pow(x(2), 2) * pow(x(0), 2);
    B_x3(0, 50) = 3 * pow(x(2), 2) * pow(x(1), 2);
    B_x3(0, 51) = 3 * pow(x(2), 2) * pow(x(3), 2);
    B_x3(0, 52) = 0.0;
    B_x3(0, 53) = 0.0;
    B_x3(0, 54) = 2 * pow(x(3), 3) * x(2);
    B_x3(0, 55) = 0.0;
    B_x3(0, 56) = 0.0;
    B_x3(0, 57) = 4 * pow(x(2), 3);
    B_x3(0, 58) = 0.0;
    B_x3(0, 59) = 0.0;
    B_x3(0, 60) = pow(x(0), 4);
    B_x3(0, 61) = 0.0;
    B_x3(0, 62) = 0.0;
    B_x3(0, 63) = pow(x(1), 4);
    B_x3(0, 64) = 0.0;
    B_x3(0, 65) = 4 * pow(x(2), 3) * x(0);
    B_x3(0, 66) = 4 * pow(x(2), 3) * x(1);
    B_x3(0, 67) = 4 * pow(x(2), 3) * x(3);
    B_x3(0, 68) = 0.0;
    B_x3(0, 69) = 0.0;
    B_x3(0, 70) = pow(x(3), 4);
    B_x3(0, 71) = 0.0;
    B_x3(0, 72) = 0.0;
    B_x3(0, 73) = 5 * pow(x(2), 4);
    B_x3(0, 74) = 0.0;

    return B_x3;
  }

  MatrixXd Controller::grad_basis_x4(VectorXd x)
  {
    MatrixXd B_x4;
    B_x4.setZero(1, 75);

    B_x4(0, 0) = 0.0;
    B_x4(0, 1) = 0.0;
    B_x4(0, 2) = 0.0;
    B_x4(0, 3) = 0.0;
    B_x4(0, 4) = 1.0;
    B_x4(0, 5) = 0.0;
    B_x4(0, 6) = 0.0;
    B_x4(0, 7) = x(3);
    B_x4(0, 8) = 0.0;
    B_x4(0, 9) = x(1);
    B_x4(0, 10) = x(2);
    B_x4(0, 11) = 0.0;
    B_x4(0, 12) = 0.0;
    B_x4(0, 13) = B_x4(0, 0) = 0.0;
    B_x4(0, 14) = 2 * x(3);
    B_x4(0, 15) = 0.0;
    B_x4(0, 16) = 0.0;
    B_x4(0, 17) = pow(x(0), 2);
    B_x4(0, 18) = 0.0;
    B_x4(0, 19) = 0.0;
    B_x4(0, 20) = pow(x(1), 2);
    B_x4(0, 21) = 0.0;
    B_x4(0, 22) = 0.0;
    B_x4(0, 23) = pow(x(2), 2);
    B_x4(0, 24) = 2 * x(3) * x(0);
    B_x4(0, 25) = 2 * x(3) * x(1);
    B_x4(0, 26) = 2 * x(3) * x(2);
    B_x4(0, 27) = 0.0;
    B_x4(0, 28) = 0.0;
    B_x4(0, 29) = 0.0;
    B_x4(0, 30) = 3 * pow(x(3), 2);
    B_x4(0, 31) = 0.0;
    B_x4(0, 32) = 0.0;
    B_x4(0, 33) = pow(x(0), 3);
    B_x4(0, 34) = 0.0;
    B_x4(0, 35) = 0.0;
    B_x4(0, 36) = pow(x(1), 3);
    B_x4(0, 37) = 0.0;
    B_x4(0, 38) = 0.0;
    B_x4(0, 39) = pow(x(2), 3);
    B_x4(0, 40) = 3 * pow(x(3), 2) * x(0);
    B_x4(0, 41) = 3 * pow(x(3), 2) * x(1);
    B_x4(0, 42) = 3 * pow(x(3), 2) * x(2);
    B_x4(0, 43) = 0.0;
    B_x4(0, 44) = 0.0;
    B_x4(0, 45) = 2 * pow(x(0), 3) * x(3);
    B_x4(0, 46) = 0.0;
    B_x4(0, 47) = 0.0;
    B_x4(0, 48) = 2 * pow(x(1), 3) * x(3);
    B_x4(0, 49) = 0.0;
    B_x4(0, 50) = 0.0;
    B_x4(0, 51) = 2 * pow(x(2), 3) * x(3);
    B_x4(0, 52) = 3 * pow(x(3), 2) * pow(x(0), 2);
    B_x4(0, 53) = 3 * pow(x(3), 2) * pow(x(1), 2);
    B_x4(0, 54) = 3 * pow(x(3), 2) * pow(x(2), 2);
    B_x4(0, 55) = 0.0;
    B_x4(0, 56) = 0.0;
    B_x4(0, 57) = 0.0;
    B_x4(0, 58) = 4 * pow(x(3), 3);
    B_x4(0, 59) = 0.0;
    B_x4(0, 60) = 0.0;
    B_x4(0, 61) = pow(x(0), 4);
    B_x4(0, 62) = 0.0;
    B_x4(0, 63) = 0.0;
    B_x4(0, 64) = pow(x(1), 4);
    B_x4(0, 65) = 0.0;
    B_x4(0, 66) = 0.0;
    B_x4(0, 67) = pow(x(2), 4);
    B_x4(0, 68) = 4 * pow(x(3), 3) * x(0);
    B_x4(0, 69) = 4 * pow(x(3), 3) * x(1);
    B_x4(0, 70) = 4 * pow(x(3), 3) * x(2);
    B_x4(0, 71) = 0.0;
    B_x4(0, 72) = 0.0;
    B_x4(0, 73) = 0.0;
    B_x4(0, 74) = 5 * pow(x(3), 4);

    return B_x4;
  }

  // Camera-UAV Velocity Transform VelUAV
  MatrixXd Controller::VelTrans(MatrixXd CameraVel)
  {
    Matrix<double, 3, 1> tt;
    tt(0, 0) = 0;
    tt(1, 0) = 0;
    tt(2, 0) = 0;

    Matrix<double, 3, 3> Tt;
    Tt(0, 0) = 0;
    Tt(0, 1) = -tt(2, 0);
    Tt(0, 2) = tt(1, 0);
    Tt(1, 0) = tt(2, 0);
    Tt(1, 1) = 0;
    Tt(1, 2) = -tt(0, 0);
    Tt(2, 0) = -tt(1, 0);
    Tt(2, 1) = tt(0, 0);
    Tt(2, 2) = 0;

    double thx = M_PI_2;
    double thy = M_PI;
    double thz = M_PI_2;

    Matrix<double, 3, 3> Rx;
    Rx(0, 0) = 1;
    Rx(0, 1) = 0;
    Rx(0, 2) = 0;
    Rx(1, 0) = 0;
    Rx(1, 1) = cos(thx);
    Rx(1, 2) = -sin(thx);
    Rx(2, 0) = 0;
    Rx(2, 1) = sin(thx);
    Rx(2, 2) = cos(thx);

    Matrix<double, 3, 3> Ry;
    Ry(0, 0) = cos(thy);
    Ry(0, 1) = 0;
    Ry(0, 2) = sin(thy);
    Ry(1, 0) = 0;
    Ry(1, 1) = 1;
    Ry(1, 2) = 0;
    Ry(2, 0) = -sin(thy);
    Ry(2, 1) = 0;
    Ry(2, 2) = cos(thy);

    Matrix<double, 3, 3> Rz;
    Rz(0, 0) = cos(thz);
    Rz(0, 1) = -sin(thz);
    Rz(0, 2) = 0;
    Rz(1, 0) = sin(thz);
    Rz(1, 1) = cos(thz);
    Rz(1, 2) = 0;
    Rz(2, 0) = 0;
    Rz(2, 1) = 0;
    Rz(2, 2) = 1;

    Matrix<double, 3, 3> Rth;
    Rth.setZero(3, 3);
    Rth = Rz * Ry * Rx;

    Matrix<double, 6, 1> VelCam;
    VelCam(0, 0) = CameraVel(0, 0);
    VelCam(1, 0) = CameraVel(1, 0);
    VelCam(2, 0) = CameraVel(2, 0);
    VelCam(3, 0) = 0;
    VelCam(4, 0) = 0;
    VelCam(5, 0) = CameraVel(3, 0);

    Matrix<double, 3, 3> Zeroes;
    Zeroes.setZero(3, 3);

    Matrix<double, 6, 6> Vtrans;
    Vtrans.block(0, 0, 3, 3) = Rth;
    Vtrans.block(0, 3, 3, 3) = Tt * Rth;
    Vtrans.block(3, 0, 3, 3) = Zeroes;
    Vtrans.block(3, 3, 3, 3) = Rth;

    Matrix<double, 6, 1> VelUAV;
    VelUAV.setZero(6, 1);
    VelUAV = Vtrans * VelCam;

    return VelUAV;
  }

  // Camera-UAV Velocity Transform VelUAV
  MatrixXd Controller::VelTrans1(MatrixXd CameraVel1)
  {
    Matrix<double, 3, 1> tt1;
    tt1(0, 0) = 0;
    tt1(1, 0) = 0;
    tt1(2, 0) = -0.14;

    Matrix<double, 3, 3> Tt1;
    Tt1(0, 0) = 0;
    Tt1(0, 1) = -tt1(2, 0);
    Tt1(0, 2) = tt1(1, 0);
    Tt1(1, 0) = tt1(2, 0);
    Tt1(1, 1) = 0;
    Tt1(1, 2) = -tt1(0, 0);
    Tt1(2, 0) = -tt1(1, 0);
    Tt1(2, 1) = tt1(0, 0);
    Tt1(2, 2) = 0;

    double thx1 = 0;
    double thy1 = M_PI_2;
    double thz1 = 0;

    Matrix<double, 3, 3> Rx1;
    Rx1(0, 0) = 1;
    Rx1(0, 1) = 0;
    Rx1(0, 2) = 0;
    Rx1(1, 0) = 0;
    Rx1(1, 1) = cos(thx1);
    Rx1(1, 2) = -sin(thx1);
    Rx1(2, 0) = 0;
    Rx1(2, 1) = sin(thx1);
    Rx1(2, 2) = cos(thx1);

    Matrix<double, 3, 3> Ry1;
    Ry1(0, 0) = cos(thy1);
    Ry1(0, 1) = 0;
    Ry1(0, 2) = sin(thy1);
    Ry1(1, 0) = 0;
    Ry1(1, 1) = 1;
    Ry1(1, 2) = 0;
    Ry1(2, 0) = -sin(thy1);
    Ry1(2, 1) = 0;
    Ry1(2, 2) = cos(thy1);

    Matrix<double, 3, 3> Rz1;
    Rz1(0, 0) = cos(thz1);
    Rz1(0, 1) = -sin(thz1);
    Rz1(0, 2) = 0;
    Rz1(1, 0) = sin(thz1);
    Rz1(1, 1) = cos(thz1);
    Rz1(1, 2) = 0;
    Rz1(2, 0) = 0;
    Rz1(2, 1) = 0;
    Rz1(2, 2) = 1;

    Matrix<double, 3, 3> Rth1;
    Rth1.setZero(3, 3);
    Rth1 = Rz1 * Ry1 * Rx1;

    Matrix<double, 6, 1> VelCam1;
    VelCam1(0, 0) = CameraVel1(0, 0);
    VelCam1(1, 0) = CameraVel1(1, 0);
    VelCam1(2, 0) = CameraVel1(2, 0);
    VelCam1(3, 0) = CameraVel1(3, 0);
    VelCam1(4, 0) = CameraVel1(4, 0);
    VelCam1(5, 0) = CameraVel1(5, 0);

    Matrix<double, 3, 3> Zeroes1;
    Zeroes1.setZero(3, 3);

    Matrix<double, 6, 6> Vtrans1;
    Vtrans1.block(0, 0, 3, 3) = Rth1;
    Vtrans1.block(0, 3, 3, 3) = Tt1 * Rth1;
    Vtrans1.block(3, 0, 3, 3) = Zeroes1;
    Vtrans1.block(3, 3, 3, 3) = Rth1;

    Matrix<double, 6, 1> VelUAV1;
    VelUAV1.setZero(6, 1);
    VelUAV1 = Vtrans1 * VelCam1;

    return VelUAV1;
  }

  MatrixXd Controller::weights_loading(string filename)
  {
    ifstream file(filename);

    if (!file.is_open())
    {
      cerr << "Error opening file!" << endl;
    }

    vector<vector<double>> data;

    string line;
    while (getline(file, line))
    {
      istringstream iss(line);
      vector<double> row;

      string value;
      while (getline(iss, value, ','))
      {
        row.push_back(stod(value));
      }

      data.push_back(row);
    }

    // Convert the vector of vectors to an Eigen MatrixXd
    MatrixXd trained_weights(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i)
    {
      for (size_t j = 0; j < data[i].size(); ++j)
      {
        trained_weights(i, j) = data[i][j];
      }
    }
    return trained_weights;
  }

  // ****UPDATE IMAGE FEATURE COORDINATES****//
  void Controller::featureCallback_poly_custom(const img_seg_cnn::POLYcalc_custom::ConstPtr &s_message)
  {
    feature_vector.setZero(s_message->features.size());
    polygon_features.setZero(s_message->features.size() / 2, 2);

    for (int i = 0; i < s_message->features.size() - 1; i += 2)
    {
      feature_vector[i] = s_message->features[i];
      feature_vector[i + 1] = s_message->features[i + 1];
    }

    for (int i = 0, j = 0; i < s_message->features.size() - 1 && j < s_message->features.size() / 2; i += 2, ++j)
    {
      polygon_features(j, 0) = feature_vector[i];
      polygon_features(j, 1) = feature_vector[i + 1];
    }

    s_bar_x = s_message->barycenter_features[0];
    s_bar_y = s_message->barycenter_features[1];

    first_min_index = s_message->d;
    second_min_index = s_message->f;

    custom_sigma = s_message->custom_sigma;
    custom_sigma_square = s_message->custom_sigma_square;
    custom_sigma_square_log = s_message->custom_sigma_square_log;

    angle_tangent = s_message->tangent;
    angle_radian = s_message->angle_radian;
    angle_deg = s_message->angle_deg;

    flag = 1;
    // cout << "(not transformed) Feature callback flag: " << flag << endl;
  }

  // //****UPDATE IMAGE FEATURE COORDINATES****//
  void Controller::featureCallback_poly_custom_tf(const img_seg_cnn::POLYcalc_custom_tf::ConstPtr &s_message)
  {
    // cout << "~~~~~~~~~~ featureCallback_poly_custom_tf ~~~~~~~~~~" << endl;
    int N = s_message->transformed_features.size();
    transformed_features.setZero(N);
    transformed_polygon_features.setZero(N / 2, 2);

    for (int i = 0; i < N - 1; i += 2)
    {
      // cout << "i = " << i << endl;
      transformed_features[i] = s_message->transformed_features[i];
      transformed_features[i + 1] = s_message->transformed_features[i + 1];
    }

    for (int i = 0, j = 0; i < N - 1 && j < N / 2; i += 2, ++j)
    {
      // cout << "i = " << i << endl;
      transformed_polygon_features(j, 0) = transformed_features[i];
      transformed_polygon_features(j, 1) = transformed_features[i + 1];
    }

    transformed_s_bar_x = s_message->transformed_barycenter_features[0];
    transformed_s_bar_y = s_message->transformed_barycenter_features[1];

    transformed_first_min_index = s_message->d_transformed;
    transformed_second_min_index = s_message->f_transformed;

    transformed_sigma = s_message->transformed_sigma;
    transformed_sigma_square = s_message->transformed_sigma_square;
    transformed_sigma_square_log = s_message->transformed_sigma_square_log;

    transformed_tangent = s_message->transformed_tangent;
    transformed_angle_radian = s_message->transformed_angle_radian;
    transformed_angle_deg = s_message->transformed_angle_deg;

    opencv_moments.setZero(s_message->moments.size());
    for (int i = 0; i < s_message->moments.size(); i++)
    {
      // cout << "i = " << i << endl;
      opencv_moments[i] = s_message->moments[i];
    }
    cX = opencv_moments[1] / opencv_moments[0];
    cY = opencv_moments[2] / opencv_moments[0];

    cX_int = (int)cX;
    cY_int = (int)cY;

    flag = 1;
    // cout << "Feature callback flag: " << flag << endl;
  }

  //****UPDATE ALTITUDE****//
  void Controller::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
  {
    Z0 = alt_message->data;
    Z1 = alt_message->data;
    Z2 = alt_message->data;
    Z3 = alt_message->data;
    flag = 1;
    // cout << "flag = " << flag << endl;
  }

  void Controller::update()
  {
    //****SEND VELOCITIES TO AUTOPILOT THROUGH MAVROS****//
    mavros_msgs::PositionTarget dataMsg;
    dataMsg.coordinate_frame = 8;
    dataMsg.type_mask = 1479;
    dataMsg.header.stamp = ros::Time::now();

    // Τracking tuning
    dataMsg.velocity.x = 0.0;
    dataMsg.velocity.y = 0.0;
    dataMsg.velocity.z = 0.0;
    dataMsg.yaw_rate = 0.0;

    while (ros::ok())
    {
      state_vector.setZero(4);
      state_vector_des.setZero(4);
      cmd_vel.setZero(4);
      velocities.setZero(4);
      error.setZero(4);
      gains.setIdentity(4, 4);

      double R = 1.0;

      MatrixXd grad_weights;
      grad_weights.setZero(1, 4);

      loaded_weights.setZero(75, 40);
      string flnm;
      // flnm = "/home/sotiris/catkin_ws/src/vs_reactive_control_uav/vs_reactive_control_controller/src/stored_weights_version_7_21_07_2023.csv";
      flnm = "/home/sotiris/controllers_catkin_ws/src/vs_reactive_control_uav/vs_reactive_control_controller/src/stored_weights_version_2_20_10_2023.csv";
      // flnm = "/home/sotiris/controllers_catkin_ws/src/vs_reactive_control_uav/vs_reactive_control_controller/src/stored_weights_version_4_20_10_2023.csv";
      // flnm = "/home/sotiris/controllers_catkin_ws/src/vs_reactive_control_uav/vs_reactive_control_controller/src/stored_weights_version_2_03_11_2023.csv";

      if (cX != 0 && cY != 0 && Z0 != 0 && Z1 != 0 && Z2 != 0 && Z3 != 0)
      {
        // state_vector << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
        state_vector << transformed_s_bar_x, transformed_s_bar_y, transformed_sigma_square_log, transformed_tangent;
        // state_vector_des << 0.0, 0.0, 5.0, angle_des_tan;
        state_vector_des << 0.0, 0.0, sigma_log_des, 0.0;

        // Check for NaN values in state vectors
        if (state_vector.hasNaN() || state_vector_des.hasNaN())
        {
          cerr << "Error: NaN values in state vectors." << endl;
          cerr << "Location: Before velocities calculation." << endl;
          exit(1); // Exit the function or return an error code as needed
        }

        error = state_vector - state_vector_des;
        cout << "state_vector = " << state_vector.transpose() << endl;
        cout << "state_vector_des = " << state_vector_des.transpose() << endl;
        // cout << "error " << error.transpose() << endl;

        if ((transformed_first_min_index == 0 && transformed_second_min_index == transformed_features.size() / 2 - 1) || (transformed_first_min_index == transformed_features.size() / 2 - 1 && transformed_second_min_index == 0))
        {
          MatrixXd model = Controller::Dynamics(transformed_features, transformed_first_min_index, transformed_second_min_index);
          // cout << "model calculated!" << endl;
          // cout << "model = \n"
          //      << model << endl;

          // MatrixXd grad_x1 = Controller::grad_basis_x1(state_vector);
          // MatrixXd grad_x2 = Controller::grad_basis_x2(state_vector);
          // MatrixXd grad_x3 = Controller::grad_basis_x3(state_vector);
          // MatrixXd grad_x4 = Controller::grad_basis_x4(state_vector);
          MatrixXd grad_x1 = Controller::grad_basis_x1(error);
          MatrixXd grad_x2 = Controller::grad_basis_x2(error);
          MatrixXd grad_x3 = Controller::grad_basis_x3(error);
          MatrixXd grad_x4 = Controller::grad_basis_x4(error);

          loaded_weights = Controller::weights_loading(flnm);
          grad_weights << grad_x1 * loaded_weights.col(30), grad_x2 * loaded_weights.col(30), grad_x3 * loaded_weights.col(30), grad_x4 * loaded_weights.col(30);

          // Check for NaN values in matrices
          if (model.hasNaN() || grad_x1.hasNaN() || grad_x2.hasNaN() || grad_x3.hasNaN() || grad_x4.hasNaN() || grad_weights.hasNaN())
          {
            cerr << "Error: NaN values in matrices." << endl;
            cerr << "Location: Before velocities calculation." << endl;
            exit(1); // Exit the function or return an error code as needed
          }

          velocities = -0.5 * R * model.transpose() * grad_weights.transpose();

          // Check for NaN values in velocities
          if (isnan(velocities.sum()))
          {
            cerr << "Error: NaN values in velocities." << endl;
            cerr << "Location: After velocities calculation." << endl;
            exit(1); // Exit the function or return an error code as needed
          }
        }
      }

      //****SEND VELOCITIES TO AUTOPILOT THROUGH MAVROS****//
      Matrix<double, 4, 1> caminputs;
      caminputs(0, 0) = velocities[0];
      caminputs(1, 0) = velocities[1];
      caminputs(2, 0) = velocities[2];
      caminputs(3, 0) = velocities[3];

      Tx = VelTrans1(VelTrans(caminputs))(0, 0);
      Ty = VelTrans1(VelTrans(caminputs))(1, 0);
      Tz = VelTrans1(VelTrans(caminputs))(2, 0);
      Oz = VelTrans1(VelTrans(caminputs))(5, 0);

      dataMsg.velocity.x = gain_tx * Tx + forward_term;
      dataMsg.velocity.y = gain_ty * Ty;
      dataMsg.velocity.z = gain_tz * Tz;
      dataMsg.yaw_rate = gain_yaw * Oz;

      printf("Drone Velocities before failsafe Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
      cout << "\n"
           << endl;

      if (dataMsg.velocity.x >= 1.0)
      {
        dataMsg.velocity.x = 0.75;
      }
      if (dataMsg.velocity.x <= -1.0)
      {
        dataMsg.velocity.x = -0.75;
      }
      if (dataMsg.velocity.y >= 0.5)
      {
        dataMsg.velocity.y = 0.4;
      }
      if (dataMsg.velocity.y <= -0.4)
      {
        dataMsg.velocity.y = -0.4;
      }
      if (dataMsg.yaw_rate >= 0.3)
      {
        dataMsg.yaw_rate = 0.2;
      }
      if (dataMsg.yaw_rate <= -0.3)
      {
        dataMsg.yaw_rate = -0.2;
      }

      printf("Final Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
      cout << "\n"
           << endl;

      std_msgs::Float64MultiArray state_vecMsg;
      for (int i = 0; i < state_vector.size(); i++)
      {
        state_vecMsg.data.push_back(state_vector[i]);
      }

      std_msgs::Float64MultiArray state_vec_desMsg;
      for (int i = 0; i < state_vector_des.size(); i++)
      {
        state_vec_desMsg.data.push_back(state_vector_des[i]);
      }

      state_vec_pub_.publish(state_vecMsg);
      state_vec_des_pub_.publish(state_vec_desMsg);
      vel_pub_.publish(dataMsg);
    }
  }

} // namespace