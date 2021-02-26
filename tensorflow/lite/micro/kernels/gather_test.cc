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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Hard coded dimension limit. Is there a predefined constant?
constexpr int MaxDims = 254;

template <typename T>
void TestGather(const int* input_dims, const T* input_data,
                    const int* positions_dims, const int32_t* positions_data,
                    const int* output_dims, const T* expected_output_data,
                    T* output_data) {
  TfLiteIntArray* in_dims = IntArrayFromInts(input_dims);
  TfLiteIntArray* pos_dims = IntArrayFromInts(positions_dims);
  TfLiteIntArray* out_dims = IntArrayFromInts(output_dims);
  const int in_dims_size = in_dims->size;
  const int out_dims_size = out_dims->size;
  const int output_size = ElementCount(*out_dims);

  // Running the op will update output_dims[], so we need to save a copy first.
  // We also need to skip output_dims[0], which is the dimension size.
  int expected_out_dims[MaxDims];
  for (int i = 0; i < out_dims_size; ++i) {
    expected_out_dims[i] = output_dims[i + 1];
  }

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, in_dims),
      CreateTensor(positions_data, pos_dims),
      CreateTensor(output_data, out_dims, true),
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_GATHER(); //TODO built-in data
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // Get the actual output tensor.
  TfLiteTensor* actual_out_tensor = &tensors[2];
  TfLiteIntArray* actual_out_dims = actual_out_tensor->dims;
  const int actual_out_dims_size = actual_out_dims->size;
  TF_LITE_MICRO_EXPECT_EQ(actual_out_dims_size, (in_dims_size + 1));
  for (int i = 0; i < actual_out_dims_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_out_dims[i], actual_out_dims->data[i]);
  }
  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest0) {
  int8_t output_data[4];
  const int input_dims[] = {2, 2, 2};
  const int8_t input_data[] = {-1, 1, -2, 2};
  const int8_t golden_data[] = {-1, 1, -2, 2};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {0};
  const int golden_dims[] = {3, 1, 2, 2};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, positions_dims,
                                          positions_data, golden_dims, golden_data,
                                          output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest1) {
  float output_data[4];
  const int input_dims[] = {2, 2, 2};
  const float input_data[] = {-1.1, 1.2, -2.1, 2.2};
  const float golden_data[] = {-1.1, 1.2, -2.1, 2.2};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {1};
  const int golden_dims[] = {3, 2, 1, 2};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, positions_dims,
                                         positions_data, golden_dims, golden_data,
                                         output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsPositiveAxisTest2) {
  int8_t output_data[4];
  const int input_dims[] = {2, 2, 2};
  const int8_t input_data[] = {-1, 1, -2, 2};
  const int8_t golden_data[] = {-1, 1, -2, 2};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {2};
  const int golden_dims[] = {3, 2, 2, 1};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, positions_dims,
                                          positions_data, golden_dims, golden_data,
                                          output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest4) {
  int8_t output_data[6];
  const int input_dims[] = {3, 3, 1, 2};
  const int8_t input_data[] = {-1, 1, 2, -2, 0, 3};
  const int8_t golden_data[] = {-1, 1, 2, -2, 0, 3};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {-4};
  const int golden_dims[] = {4, 1, 3, 1, 2};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, positions_dims,
                                          positions_data, golden_dims, golden_data,
                                          output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest3) {
  float output_data[6];
  const int input_dims[] = {3, 3, 1, 2};
  const float input_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const float golden_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {-3};
  const int golden_dims[] = {4, 3, 1, 1, 2};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, positions_dims,
                                         positions_data, golden_dims, golden_data,
                                         output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest2) {
  int8_t output_data[6];
  const int input_dims[] = {3, 1, 2, 3};
  const int8_t input_data[] = {-1, 1, 2, -2, 0, 3};
  const int8_t golden_data[] = {-1, 1, 2, -2, 0, 3};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {-2};
  const int golden_dims[] = {4, 1, 2, 1, 3};
  tflite::testing::TestExpandDims<int8_t>(input_dims, input_data, positions_dims,
                                          positions_data, golden_dims, golden_data,
                                          output_data);
}

TF_LITE_MICRO_TEST(ExpandDimsNegativeAxisTest1) {
  float output_data[6];
  const int input_dims[] = {3, 1, 3, 2};
  const float input_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const float golden_data[] = {0.1, -0.8, -1.2, -0.5, 0.9, 1.3};
  const int positions_dims[] = {1, 1};
  const int32_t positions_data[] = {-1};
  const int golden_dims[] = {4, 1, 3, 2, 1};
  tflite::testing::TestExpandDims<float>(input_dims, input_data, positions_dims,
                                         positions_data, golden_dims, golden_data,
                                         output_data);
}

TF_LITE_MICRO_TESTS_END
