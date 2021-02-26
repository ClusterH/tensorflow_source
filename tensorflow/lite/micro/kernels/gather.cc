/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (positions->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Positions of type '%s' are not supported by gather.",
                         TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  int axis = params->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  const int num_dimensions =
      NumDimensions(input) + NumDimensions(positions) - 1;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = 0; i < positions->dims->size; ++i) {
    output_shape->data[output_index++] = positions->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  return kTfLiteOk;
}

template <typename InputT, typename PositionsT>
TfLiteStatus Gather(const TfLiteGatherParams& params, const TfLiteTensor* input,
                    const TfLiteTensor* positions, TfLiteTensor* output) {
  tflite::GatherParams op_params;
  op_params.axis = params.axis;
  optimized_ops::Gather(op_params, GetTensorShape(input),
                        GetTensorData<InputT>(input), GetTensorShape(positions),
                        GetTensorData<PositionsT>(positions),
                        GetTensorShape(output), GetTensorData<InputT>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* positions;
      tflite::micro::GetEvalInput(context, node, kInputPositions);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (positions->type == kTfLiteInt32) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int32_t>(params, input, positions, output);
      case kTfLiteInt8:
        return Gather<int8_t, int32_t>(params, input, positions, output);
      default:
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  if (positions->type == kTfLiteInt64) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int64_t>(params, input, positions, output);
      case kTfLiteInt8:
        return Gather<int8_t, int64_t>(params, input, positions, output);
      default:
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  TF_LITE_KERNEL_LOG(context,
                     "Positions of type '%s' are not supported by gather.",
                     TfLiteTypeGetName(positions->type));
  return kTfLiteError;
}
}  // namespace

TfLiteRegistration Register_GATHER() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
