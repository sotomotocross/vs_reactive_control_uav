#include "vs_reactive_control_controller/WeightLoader.hpp"

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
  Eigen::MatrixXd WeightLoader::weights_loading(std::string filename)
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
} // namespace vs_reactive_control_controller
