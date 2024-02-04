#include "vs_reactive_control_controller/UtilityFunctions.hpp"

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
  // Function for transforming camera velocities to UAV velocities for Camera 1
  Eigen::MatrixXd UtilityFunctions::VelTrans1(Eigen::MatrixXd CameraVel1)
  {
    // Transformation matrices for Camera
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

    // Conversion of camera velocities to UAV velocities
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

  Eigen::MatrixXd UtilityFunctions::VelTrans(Eigen::MatrixXd CameraVel)
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

  Eigen::MatrixXd UtilityFunctions::Dynamics(Eigen::VectorXd feat_prop,
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

  Eigen::MatrixXd UtilityFunctions::grad_basis_x1(Eigen::VectorXd x)
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

  Eigen::MatrixXd UtilityFunctions::grad_basis_x2(Eigen::VectorXd x)
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

  Eigen::MatrixXd UtilityFunctions::grad_basis_x3(Eigen::VectorXd x)
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

  Eigen::MatrixXd UtilityFunctions::grad_basis_x4(Eigen::VectorXd x)
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

  Eigen::MatrixXd UtilityFunctions::weights_loading(std::string filename)
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

  // Add implementations for other utility functions...
} // namespace
