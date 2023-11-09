#include "vs_reactive_control_controller/controller.hpp"

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
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

namespace vs_reactive_control_controller
{

  Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
  {
    // parameters
    pnh_.param<double>("controller/update_frequency", update_frequency_, 30.0);
    pnh_.param<double>("controller/k_p", k_p_, 0.01);
    pnh_.param<double>("controller/k_d", k_d_, 0.01);
    pnh_.param<double>("controller/max_vel_cmd", max_vel_cmd_, 4.0);
    pnh_.param<double>("controller/test_yaw_rate", test_yaw_rate_, 0.0);

    pnh_.param<double>("line/theta_upper", theta_upper_, 0.3);
    pnh_.param<double>("line/theta_lower", theta_lower_, -0.3);
    pnh_.param<double>("line/line_vel_threshold", line_vel_threshold_, 4.0);
    pnh_.param<double>("line/line_min_length", line_min_length_, 40.0);

    pnh_.param<double>("reactive_controller/gain_tx", gain_tx, 1.0);
    pnh_.param<double>("reactive_controller/gain_ty", gain_ty, 1.0);
    pnh_.param<double>("reactive_controller/gain_tz", gain_tz, 2.0);
    pnh_.param<double>("reactive_controller/gain_yaw", gain_yaw, 1.0);

    line_sub_ = nh_.subscribe("lines", 1, &Controller::linesCallback, this);
    velocity_command_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("vel_cmd", 10);

    feature_sub_poly_custom_ = nh_.subscribe("polycalc_custom", 10, &Controller::featureCallback_poly_custom, this);
    feature_sub_poly_custom_tf_ = nh.subscribe("polycalc_custom_tf", 10, &Controller::featureCallback_poly_custom_tf, this);
    alt_sub_ = nh.subscribe("/mavros/global_position/rel_alt", 10, &Controller::altitudeCallback, this);

    vel_pub_ = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    // rec_pub_ = nh.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);
    cmd_vel_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/cmd_vel", 1);
    state_vec_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
    state_vec_des_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);
    img_moments_error_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/img_moments_error", 1);
    moments_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/moments", 1);
    central_moments_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/central_moments", 1);

    std::thread control_loop_thread(&Controller::update, this);
    control_loop_thread.detach();
  }

  Controller::~Controller()
  {
    velocity_command_pub_.shutdown();
  }

  void Controller::linesCallback(const vs_reactive_control_msgs::LinesConstPtr &msg)
  {
    std::unique_lock<std::mutex> lock(lines_mutex_);
    lines_ = *msg;
    last_update_ = ros::Time::now();
  }

  VectorXd Controller::Dynamics(VectorXd camTwist, VectorXd feat_prop)
  {
    MatrixXd model_mat(dim_s, dim_inputs);
    

    // Barycenter dynamics calculation
    double term_1_4 = 0.0;
    double term_1_5 = 0.0;
    double term_2_4 = 0.0;
    double term_2_5 = 0.0;

    int N;
    N = transformed_features.size() / 2;
    
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
      sum_4_4_1 = sum_4_4_1 + pow(feat_prop[i + 1], 2);
      sum_4_4_2 = sum_4_4_2 + feat_prop[i] * feat_prop[i + 1];
    }

    term_4_4_1 = transformed_tangent / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
    term_4_4_2 = (pow(y[transformed_first_min_index], 2) + pow(y[transformed_second_min_index], 2) - (2 / N) * sum_4_4_1);
    term_4_4_3 = -1 / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
    term_4_4_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_4_2);

    g_4_4 = term_4_4_1 * term_4_4_2 + term_4_4_3 * term_4_4_4;

    // Fifth term
    double term_4_5_1, term_4_5_2, term_4_5_3, term_4_5_4;
    double sum_4_5_1 = 0.0, sum_4_5_2 = 0.0;

    for (int i = 0; i < N - 1; i += 2)
    {
      sum_4_5_1 = sum_4_5_1 + pow(feat_prop[i], 2);
      sum_4_5_2 = sum_4_5_2 + feat_prop[i] * feat_prop[i + 1];
    }

    term_4_5_1 = 1 / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
    term_4_5_2 = (pow(x[transformed_first_min_index], 2) + pow(x[transformed_second_min_index], 2) - (2 / N) * sum_4_5_1);
    term_4_5_3 = -transformed_tangent / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
    term_4_5_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_5_2);

    g_4_5 = term_4_5_1 * term_4_5_2 + term_4_5_3 * term_4_5_4;

    // Fifth term
    g_4_6 = pow(transformed_tangent, 2) + 1;

    model_mat << -1 / Z0, 0.0, transformed_s_bar_x / Z0, transformed_s_bar_y,
        0.0, -1 / Z0, transformed_s_bar_y / Z0, -transformed_s_bar_x,
        0.0, 0.0, 2 / Z0, 0.0,
        0.0, 0.0, 0.0, g_4_6;

    return model_mat * camTwist;
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
    Tt(1, 0) = tt(2, 0);
    Tt(2, 0) = -tt(1, 0);
    Tt(0, 1) = -tt(2, 0);
    Tt(1, 1) = 0;
    Tt(2, 1) = tt(0, 0);
    Tt(0, 2) = tt(1, 0);
    Tt(1, 2) = -tt(0, 0);
    Tt(2, 2) = 0;
    double thx = M_PI_2;
    double thy = M_PI;
    double thz = M_PI_2;

    Matrix<double, 3, 3> Rx;
    Rx(0, 0) = 1;
    Rx(1, 0) = 0;
    Rx(2, 0) = 0;
    Rx(0, 1) = 0;
    Rx(1, 1) = cos(thx);
    Rx(2, 1) = sin(thx);
    Rx(0, 2) = 0;
    Rx(1, 2) = -sin(thx);
    Rx(2, 2) = cos(thx);

    Matrix<double, 3, 3> Ry;
    Ry(0, 0) = cos(thy);
    Ry(1, 0) = 0;
    Ry(2, 0) = -sin(thy);
    Ry(0, 1) = 0;
    Ry(1, 1) = 1;
    Ry(2, 1) = 0;
    Ry(0, 2) = sin(thy);
    Ry(1, 2) = 0;
    Ry(2, 2) = cos(thy);

    Matrix<double, 3, 3> Rz;
    Rz(0, 0) = cos(thz);
    Rz(1, 0) = sin(thz);
    Rz(2, 0) = 0;
    Rz(0, 1) = -sin(thz);
    Rz(1, 1) = cos(thz);
    Rz(2, 1) = 0;
    Rz(0, 2) = 0;
    Rz(1, 2) = 0;
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
    Tt1(1, 0) = tt1(2, 0);
    Tt1(2, 0) = -tt1(1, 0);
    Tt1(0, 1) = -tt1(2, 0);
    Tt1(1, 1) = 0;
    Tt1(2, 1) = tt1(0, 0);
    Tt1(0, 2) = tt1(1, 0);
    Tt1(1, 2) = -tt1(0, 0);
    Tt1(2, 2) = 0;

    double thx1 = 0;
    double thy1 = M_PI_2;
    double thz1 = 0;

    Matrix<double, 3, 3> Rx1;
    Rx1(0, 0) = 1;
    Rx1(1, 0) = 0;
    Rx1(2, 0) = 0;
    Rx1(0, 1) = 0;
    Rx1(1, 1) = cos(thx1);
    Rx1(2, 1) = sin(thx1);
    Rx1(0, 2) = 0;
    Rx1(1, 2) = -sin(thx1);
    Rx1(2, 2) = cos(thx1);

    Matrix<double, 3, 3> Ry1;
    Ry1(0, 0) = cos(thy1);
    Ry1(1, 0) = 0;
    Ry1(2, 0) = -sin(thy1);
    Ry1(0, 1) = 0;
    Ry1(1, 1) = 1;
    Ry1(2, 1) = 0;
    Ry1(0, 2) = sin(thy1);
    Ry1(1, 2) = 0;
    Ry1(2, 2) = cos(thy1);

    Matrix<double, 3, 3> Rz1;
    Rz1(0, 0) = cos(thz1);
    Rz1(1, 0) = sin(thz1);
    Rz1(2, 0) = 0;
    Rz1(0, 1) = -sin(thz1);
    Rz1(1, 1) = cos(thz1);
    Rz1(2, 1) = 0;
    Rz1(0, 2) = 0;
    Rz1(1, 2) = 0;
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

  //****UPDATE IMAGE FEATURE COORDINATES****//
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
    // cout << "Feature callback flag: " << flag << endl;
  }

  //****UPDATE IMAGE FEATURE COORDINATES****//
  void Controller::featureCallback_poly_custom_tf(const img_seg_cnn::POLYcalc_custom_tf::ConstPtr &s_message)
  {
    transformed_features.setZero(s_message->transformed_features.size());
    transformed_polygon_features.setZero(s_message->transformed_features.size() / 2, 2);

    for (int i = 0; i < s_message->transformed_features.size() - 1; i += 2)
    {
      transformed_features[i] = s_message->transformed_features[i];
      transformed_features[i + 1] = s_message->transformed_features[i + 1];
    }

    for (int i = 0, j = 0; i < s_message->transformed_features.size() - 1 && j < s_message->transformed_features.size() / 2; i += 2, ++j)
    {
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
    // cout << "opencv_moments before subscription: " << opencv_moments.transpose() << endl;
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
  }

  void Controller::update()
  {
    ros::Rate r(update_frequency_);

    geometry_msgs::TwistStamped vel_cmd;
    vel_cmd.header.stamp = ros::Time::now();
    vel_cmd.twist.linear.x = 0.0;
    vel_cmd.twist.linear.y = 0.0;
    vel_cmd.twist.linear.z = 0.0;

    vel_cmd.twist.angular.x = 0.0;
    vel_cmd.twist.angular.y = 0.0;
    vel_cmd.twist.angular.z = 0.0;

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

    last_vel_cmd_ = 0.0;

    while (ros::ok())
    {
      vs_reactive_control_msgs::Line longest_line;
      double max_length = 0;

      cout << "flag: " << flag << endl;
      cout << "state_vector = " << state_vector << endl;
      cout << "opencv_moments after subscription: " << opencv_moments.transpose() << endl;
      // state_vector << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
      // state_vector_des << 0.0, 0.0, 5.0, angle_des_tan;

      // cout << "((opencv_moments[1] / opencv_moments[0]) - cu) / l = " << ((opencv_moments[1] / opencv_moments[0]) - cu) / l << endl;
      // cout << "((opencv_moments[2] / opencv_moments[0]) - cu) / l = " << ((opencv_moments[2] / opencv_moments[0]) - cu) / l << endl;

      // cout << "update_frequency_ = " << update_frequency_ << endl;
      

      {
        std::unique_lock<std::mutex> lock(lines_mutex_);

        // longest line
        if ((ros::Time::now() - last_update_).toSec() * 1000 < 30)
        {
          for (auto const &line : lines_.lines)
          {
            if (line.state == vs_reactive_control_msgs::Line::INITIALIZING)
              continue;

            if (line.theta < theta_lower_ || line.theta > theta_upper_)
              continue;
            int mid_point_offset = 260 / 2 - line.pos_y;
            if (abs(mid_point_offset) > 25)
              continue;
            ;

            if (line.length > max_length)
            {
              longest_line = line;
              max_length = line.length;
            }
          }
        }
      }

      if (max_length > 0)
      {

        double yaw_vel = 0.0;
        double mean_mid_point_x = longest_line.pos_x;

        // PD controller
        double e = lines_.width / 2.0 - mean_mid_point_x;
        double time_passed = (ros::Time::now() - last_update_).toSec();
        double e_dot = (e - e_last_) / time_passed;
        if (e_dot < 0)
          e_dot = 0;
        double p_part = k_p_ * e;
        double d_part = -k_d_ * e_dot;
        yaw_vel = p_part + d_part;

        // limit yaw rate
        if (yaw_vel > max_vel_cmd_)
        {
          yaw_vel = max_vel_cmd_;
        }
        else if (yaw_vel < 0)
        {
          yaw_vel = 0;
        }

        vel_cmd.twist.angular.z = yaw_vel;
        last_vel_cmd_ = yaw_vel;
        std::cout << "LINE DETECTED: " << longest_line.id << ", VEL CMD: " << yaw_vel << ", P: " << p_part << ", D: " << d_part << std::endl;
      }
      else
      {
        vel_cmd.twist.angular.z = last_vel_cmd_;
        std::cout << "NO LINE DETECTED, VEL CMD: " << last_vel_cmd_ << std::endl;
      }

      velocity_command_pub_.publish(vel_cmd);
      r.sleep();
    }
  }

} // namespace