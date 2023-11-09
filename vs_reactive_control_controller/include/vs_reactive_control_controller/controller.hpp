#pragma once

#include <ros/ros.h>

#include <mutex>
#include <vs_reactive_control_msgs/Lines.h>

#include <thread>
#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace vs_reactive_control_controller
{

  /**
   * @brief The controller class used for HW test
   * this controller tracks a spinning line by calculating
   * yaw angular velocity commands
   */
  class Controller
  {

  public:
    Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh);
    ~Controller();

    VectorXd state_vector;
    VectorXd state_vector_des;
    VectorXd cmd_vel;
    VectorXd error;
    VectorXd barx_meas;
    VectorXd barx_des;

  private:
    /**
     * @brief ros lines callback, draws lines and their ids into the current image
     */
    void linesCallback(const vs_reactive_control_msgs::LinesConstPtr &msg);

    void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message);

    void featureCallback_poly_custom_tf(const img_seg_cnn::POLYcalc_custom_tf::ConstPtr &s_message);

    void featureCallback_poly_custom(const img_seg_cnn::POLYcalc_custom::ConstPtr &s_message);

    MatrixXd VelTrans1(MatrixXd CameraVel1);
    MatrixXd VelTrans(MatrixXd CameraVel);

    VectorXd Dynamics(VectorXd camTwist, VectorXd feat_prop);

    /**
     * @brief controller update, calculated vel cmd and publishes it
     */
    void update();

  private:
    // ros
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber line_sub_;
    ros::Publisher velocity_command_pub_;

    ros::Subscriber feature_sub_poly_custom_;
    ros::Subscriber feature_sub_poly_custom_tf_;
    ros::Subscriber alt_sub_;

    ros::Publisher vel_pub_;
    ros::Publisher rec_pub_;
    ros::Publisher cmd_vel_pub_ ;
    ros::Publisher state_vec_pub_ ;
    ros::Publisher state_vec_des_pub_ ;
    ros::Publisher img_moments_error_pub_ ;
    ros::Publisher moments_pub_ ;
    ros::Publisher central_moments_pub_ ;

    // PD controller
    double update_frequency_;
    double k_p_;
    double k_d_;
    double max_vel_cmd_;
    double e_last_ = 0.0;

    const int dim_inputs = 4;
    int dim_s = 4;

    double cX, cY;
    int cX_int, cY_int;

    // Simulator camera parameters
    double umax = 720;
    double umin = 0;
    double vmax = 480;
    double vmin = 0;
    double cu = 360.5;
    double cv = 240.5;
    double l = 252.07;

    double sc_x = (umax - cu) / l;
    double sc_y = (vmax - cv) / l;

    int flag = 0;

    // Camera Frame Update Callback Variables
    double Z0, Z1, Z2, Z3;
    double Tx, Ty, Tz, Oz;

    double s_bar_x, s_bar_y;
    double first_min_index, second_min_index;
    double custom_sigma, custom_sigma_square, custom_sigma_square_log;
    double angle_tangent, angle_radian, angle_deg;

    double transformed_s_bar_x, transformed_s_bar_y;
    double transformed_first_min_index, transformed_second_min_index;
    double transformed_sigma, transformed_sigma_square, transformed_sigma_square_log;
    double transformed_tangent, transformed_angle_radian, transformed_angle_deg;

    double s_bar_x_des = Z0 * (cu - cu / l);
    double s_bar_y_des = Z0 * (cv - cv / l);

    // double sigma_des = 18500.0;  // Value for outdoors experiments
    double sigma_des = 18.5;
    double sigma_square_des = sqrt(sigma_des);
    double sigma_log_des = log(sigma_square_des);

    double angle_deg_des = 0;
    double angle_des_tan = tan((angle_deg_des / 180) * 3.14);

    // TESTING
    double test_yaw_rate_;

    // line selection
    double theta_upper_;
    double theta_lower_;
    double line_vel_threshold_;
    double line_min_length_;

    MatrixXd gains;
    VectorXd feature_vector;
    VectorXd transformed_features;
    VectorXd opencv_moments;
    MatrixXd polygon_features;
    MatrixXd transformed_polygon_features;

    VectorXd feat_u_vector;
    VectorXd feat_v_vector;
    VectorXd feat_vector;

    double gain_tx;
    double gain_ty;
    double gain_tz;
    double gain_yaw;

    std::mutex lines_mutex_;

    vs_reactive_control_msgs::Lines lines_;
    ros::Time last_update_;
    ros::Time last_line_stamp_;

    // TESTING
    double last_vel_cmd_;
  };

} // namespace