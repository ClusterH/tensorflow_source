/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"

#include <string>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, tensorflow_batch_padding_policy, "PAD_UP",
          "The policy that a batch schduler is using when deciding what to do "
          "when, say, 18 requests need to be batched, but only 16 and 32 batch "
          "sizes are allowed. The following options are available. PAD_UP: pad "
          "to size 32. BATCH_DOWN: schedule a batch of size 16 and leave 2 "
          "requests in the batch buffer. MINIMIZE_TPU_COST_PER_REQUEST: a "
          "smarter greedy policy that chooses to either PAD_UP or BATCH_DOWN "
          "so as to minimize the TPU costs per real request. In this case, it "
          "would compare (batch_16_cost / 16) and (batch_32_cost / 18). "
          "WARNING: not all batch schedulers might support this option.");

namespace tensorflow {
namespace serving {

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    absl::string_view attr_value) {
  if (attr_value == kLowPriorityPaddingWithMaxBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  } else if (attr_value ==
             kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
  } else if (attr_value == kPriorityIsolationAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityIsolation;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown mixed priority batching policy: %s", attr_value));
}

BatchPaddingPolicy GetBatchPaddingPolicy() {
  std::string str = absl::GetFlag(FLAGS_tensorflow_batch_padding_policy);

  if (str == "PAD_UP") {
    return BatchPaddingPolicy::kPadUp;
  } else if (str == "BATCH_DOWN") {
    return BatchPaddingPolicy::kBatchDown;
  } else if (str == "MINIMIZE_TPU_COST_PER_REQUEST") {
    return BatchPaddingPolicy::kMinimizeTpuCostPerRequest;
  } else {
    LOG(FATAL) << "Unknown enum flag value --"  // Crash ok
               << FLAGS_tensorflow_batch_padding_policy.Name() << "=" << str
               << ". Here is the flag help: "
               << FLAGS_tensorflow_batch_padding_policy.Help();
  }
}

}  // namespace serving
}  // namespace tensorflow
