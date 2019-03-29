/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/scan_ops.h"
#include "tensorflow/core/kernels/scan_ops_gpu.h"

namespace tensorflow {
using Eigen::GpuDevice;
template struct functor::Scan<GpuDevice, Eigen::internal::SumReducer<float>,
                              float>;
template struct functor::Scan<GpuDevice, Eigen::internal::ProdReducer<float>,
                              float>;
template struct functor::Scan<GpuDevice, Eigen::internal::SumReducer<int>,
                              int>;
template struct functor::Scan<GpuDevice, Eigen::internal::ProdReducer<int>,
                              int>;
template struct functor::Scan<GpuDevice, Eigen::internal::SumReducer<uint>,
                              uint>;
template struct functor::Scan<GpuDevice, Eigen::internal::ProdReducer<uint>,
                              uint>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
