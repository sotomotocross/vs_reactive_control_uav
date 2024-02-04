#pragma once

#include "img_seg_cnn/PolyCalcCustom.h"
#include "img_seg_cnn/PolyCalcCustomTF.h"

namespace vs_reactive_control_controller
{
  struct FeatureData
  {
    img_seg_cnn::PolyCalcCustomTF::ConstPtr poly_custom_tf_data;
    img_seg_cnn::PolyCalcCustom::ConstPtr poly_custom_data;
  };
} // namespace
