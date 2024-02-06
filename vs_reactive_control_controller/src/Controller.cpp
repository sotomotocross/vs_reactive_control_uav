#include "vs_reactive_control_controller/Controller.hpp"
#include "vs_reactive_control_controller/FeatureData.hpp"
// #include "vs_reactive_control_controller/UtilityFunctions.hpp"
#include "vs_reactive_control_controller/VelocityTransformer.hpp"
#include "vs_reactive_control_controller/DynamicsCalculator.hpp"
#include "vs_reactive_control_controller/GradientBasisCalculator.hpp"
#include "vs_reactive_control_controller/WeightLoader.hpp"

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
  // Initialize the static member of the UtilityFunctions
  DynamicsCalculator Controller::dynamics_calculator;
  
  // Constructor
  Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh)
  {
    // Initialize member variables
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

    sigma_des = 18.5;
    sigma_square_des = sqrt(sigma_des);
    sigma_log_des = log(sigma_square_des);

    angle_deg_des = 0;
    angle_des_tan = tan((angle_deg_des / 180) * 3.14);

    flag = 0;    

    // Load parameters from ROS parameter server
    pnh_.param<double>("reactive_controller/forward_term", forward_term, 1.0);
    pnh_.param<double>("reactive_controller/gain_tx", gain_tx, 1.0);
    pnh_.param<double>("reactive_controller/gain_ty", gain_ty, 1.0);
    pnh_.param<double>("reactive_controller/gain_tz", gain_tz, 2.0);
    pnh_.param<double>("reactive_controller/gain_yaw", gain_yaw, 1.0);

    // Set up ROS subscribers
    feature_sub_poly_custom_ = nh_.subscribe("polycalc_custom", 10, &Controller::featureCallback_poly_custom, this);
    feature_sub_poly_custom_tf_ = nh.subscribe("polycalc_custom_tf", 10, &Controller::featureCallback_poly_custom_tf, this);
    alt_sub_ = nh.subscribe("/mavros/global_position/rel_alt", 10, &Controller::altitudeCallback, this);

    // Set up ROS publishers
    vel_pub_ = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    state_vec_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
    state_vec_des_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);

    // Create a thread for the control loop
    control_loop_thread = std::thread([this]()
                                      {
      ros::Rate rate(50); // Adjust the rate as needed
      while (ros::ok())
      {
        update();
        rate.sleep();
      } });
  }

  // Destructor
  Controller::~Controller()
  {
    // Shutdown ROS publishers...
    vel_pub_.shutdown();
  }

  // Callback for altitude data
  void Controller::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
  {
    // Handle altitude data...
    Z0 = alt_message->data;
    Z1 = alt_message->data;
    Z2 = alt_message->data;
    Z3 = alt_message->data;
    flag = 1;
    // cout << "flag = " << flag << endl;
    cout << "Z0: " << Z0 << endl;
  }

  // Callback for custom TF features
  void Controller::featureCallback_poly_custom_tf(const img_seg_cnn::PolyCalcCustomTF::ConstPtr &s_message)
  {
    feature_data_.poly_custom_tf_data = s_message;
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

    cout << "transformed_angle_deg: " << transformed_angle_deg << endl;
    flag = 1;
    // cout << "Feature callback flag: " << flag << endl;
  }

  // Callback for custom features
  void Controller::featureCallback_poly_custom(const img_seg_cnn::PolyCalcCustom::ConstPtr &s_message)
  {
    // Handle custom features...
    feature_data_.poly_custom_data = s_message;
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

    cout << "angle_deg: " << angle_deg << endl;
    flag = 1;
    // cout << "(not transformed) Feature callback flag: " << flag << endl;
  }

  // Main update function
  void Controller::update()
  {
    // Main update logic...

    // Add print statements for debugging
    // ROS_INFO("Update function called...");
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

        cout << "Z0 = " << Z0 << endl;

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
        cout << "error " << error.transpose() << endl;

        if ((transformed_first_min_index == 0 && transformed_second_min_index == transformed_features.size() / 2 - 1) || (transformed_first_min_index == transformed_features.size() / 2 - 1 && transformed_second_min_index == 0))
        {
          // MatrixXd model = UtilityFunctions::Dynamics(transformed_features, transformed_first_min_index, transformed_second_min_index, transformed_s_bar_x, transformed_s_bar_y, transformed_tangent, Z0);
          // Call Dynamics function using the static instance
          // MatrixXd model = DynamicsCalculator::Dynamics(transformed_features, transformed_first_min_index, transformed_second_min_index, transformed_s_bar_x, transformed_s_bar_y, transformed_tangent, Z0);
          MatrixXd model = dynamics_calculator.Dynamics(transformed_features, transformed_first_min_index, transformed_second_min_index, transformed_s_bar_x, transformed_s_bar_y, transformed_tangent, Z0);
          // cout << "model calculated!" << endl;
          // cout << "model = \n"
          //      << model << endl;

          MatrixXd grad_x1 = GradientBasisCalculator::grad_basis_x1(error);
          MatrixXd grad_x2 = GradientBasisCalculator::grad_basis_x2(error);
          MatrixXd grad_x3 = GradientBasisCalculator::grad_basis_x3(error);
          MatrixXd grad_x4 = GradientBasisCalculator::grad_basis_x4(error);

          loaded_weights = WeightLoader::weights_loading(flnm);
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

      Tx = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(0, 0);
      Ty = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(1, 0);
      Tz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(2, 0);
      Oz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(5, 0);

      dataMsg.velocity.x = gain_tx * Tx + forward_term;
      dataMsg.velocity.y = gain_ty * Ty;
      dataMsg.velocity.z = gain_tz * Tz;
      dataMsg.yaw_rate = gain_yaw * Oz;

      // printf("Drone Velocities before failsafe Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
      // cout << "\n"
      //      << endl;

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

      // printf("Final Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
      // cout << "\n"
      //      << endl;

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
      // vel_pub_.publish(dataMsg);
    }
  }
} // namespace
