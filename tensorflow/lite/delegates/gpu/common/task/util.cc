/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/task/util.h"

#include <cfloat>
#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetGlslConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
  if (src_type == dst_type) {
    return "";
  }
  bool need_explicit_conversion = true;
  switch (dst_type) {
    case DataType::BOOL:
    case DataType::FLOAT32:
    case DataType::FLOAT16:
      if (gpu_info.IsGlslSupportsExplicitFp16()) {
        if (src_type == dst_type) {
          need_explicit_conversion = false;
        }
      } else {
        if (src_type == DataType::FLOAT32 || src_type == DataType::FLOAT16) {
          need_explicit_conversion = false;
        }
      }
      break;
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
      if (src_type == DataType::INT32 || src_type == DataType::INT16 ||
          src_type == DataType::INT8) {
        need_explicit_conversion = false;
      }
      break;
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
      if (src_type == DataType::UINT32 || src_type == DataType::UINT16 ||
          src_type == DataType::UINT8) {
        need_explicit_conversion = false;
      }
      break;
    default:
      break;
  }
  if (need_explicit_conversion) {
    return ToGlslShaderDataType(
        dst_type, vec_size,
        /*add_precision*/ false,
        /*explicit_fp16*/ gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}
}  // namespace

std::string MemoryTypeToCLType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "__global";
    case MemoryType::CONSTANT:
      return "__constant";
    case MemoryType::LOCAL:
      return "__local";
  }
  return "";
}

std::string MemoryTypeToMetalType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "device";
    case MemoryType::CONSTANT:
      return "constant";
      break;
    case MemoryType::LOCAL:
      return "threadgroup";
  }
  return "";
}

std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x) {
  // int p0 = src_x / batch_size;\n";
  // int b0 = src_x % batch_size;\n";
  // return p0 * stride_x * batch_size + b0 + padding_x;\n";
  return absl::Substitute("((($0) / $1) * $2 * $1 + (($0) % $1) + $3)", src_x,
                          batch_size, stride_x, padding_x);
}

std::string GetXStrideCorrectedV2(const std::string& src_x,
                                  const std::string& batch_size,
                                  const std::string& stride_x,
                                  const std::string& padding_x) {
  // int p0 = src_x / batch_size;\n";
  // int b0 = src_x % batch_size;\n";
  // return (p0 * stride_x + padding_x) * batch_size + b0;\n";
  return absl::Substitute("(((($0) / $1) * $2 + $3) * $1 + ($0) % $1)", src_x,
                          batch_size, stride_x, padding_x);
}

float4 GetMaskForLastPlane(int channels) {
  float4 mask = float4(0.0f);
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size) {
  const float task_size_per_cu =
      task_size / static_cast<float>(gpu_info.GetComputeUnitsCount());
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!gpu_info.IsMali()) {
    return 1;
  }
  MaliInfo mali_info = gpu_info.mali_info;
  switch (precision) {
    case CalculationsPrecision::F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 16.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1) {
    block_size = 1;
  } else if (task_size_per_cu <= threshold_2) {
    block_size = 2;
  } else if (task_size_per_cu <= threshold_4) {
    block_size = 4;
  } else {
    block_size = 8;
  }
  return block_size;
}

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size) {
  int3 work_groups_count;
  work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
  work_groups_count.y = DivideRoundUp(grid_size.y, work_group_size.y);
  work_groups_count.z = DivideRoundUp(grid_size.z, work_group_size.z);
  return work_groups_count;
}

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return ToCLDataType(data_type, vec_size);
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size);
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, true,
                                gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}

std::string GetZeroValue(const GpuInfo& gpu_info, DataType data_type,
                         int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(0)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(0)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(0)";
  } else {
    return "";
  }
}

std::string GetOneValue(const GpuInfo& gpu_info, DataType data_type,
                        int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(1)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(1)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(1)";
  } else {
    return "";
  }
}

std::string GetTypeConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
  if (src_type != dst_type) {
    if (gpu_info.IsApiOpenCl()) {
      return "convert_" + ToCLDataType(dst_type, vec_size) + "($0)";
    } else if (gpu_info.IsApiMetal()) {
      return dst_type == DataType::BOOL
                 ? "convert_" + ToMetalDataType(dst_type, vec_size) + "($0)"
                 : ToMetalDataType(dst_type, vec_size) + "($0)";
    } else if (gpu_info.IsGlsl()) {
      const std::string conversion =
          GetGlslConversion(gpu_info, src_type, dst_type, vec_size);
      if (!conversion.empty()) {
        return conversion + "($0)";
      } else {
        return "$0";
      }
    }
  }
  return "$0";
}

}  // namespace gpu
}  // namespace tflite
