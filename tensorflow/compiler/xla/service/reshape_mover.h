/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// This pass sinks kReshape and kTranspose operations (known as "rearrange" ops)
// down through elementwise ops:
//
//   op(rearrange(x), rearrange(y)) => rearrange(op(x, y)).
//
// We also handle the case where one of the operands is not itself a rearrange
// op but can be trivially rearranged.  For example:
//
//   op(rearrange(x), broadcast(scalar_y)) =>
//   rearrange(x, broadcast'(scalar_y)).
//
// This pass should be run to a fixed point.  It also expects algsimp to be run
// after each iteration.

struct ReshapeMoverOptions {
  // On some platforms, it's cheap to do `reshape(broadcast(f32[n] x))`.  The
  // reshape and broadcast can always be fused, and the index calculations are
  // not expensive.  In such cases it can be beneficial for us to create these
  // reshapes eagerly, allowing us to get rid of more expensive ones.
  bool reshape_of_1d_broadcast_is_cheap = false;
};

class ReshapeMover : public HloModulePass {
 public:
  explicit ReshapeMover(
      const ReshapeMoverOptions& options = ReshapeMoverOptions{})
      : options_(options) {}

  absl::string_view name() const override { return "reshape-mover"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> TryReshapeMoveOnCandidates(HloInstructionSet* candidates);
  StatusOr<bool> SinkRearrangeOperands(HloInstruction* instruction);
  StatusOr<HloInstruction*> ApplyInverseRearrange(
      const HloInstruction* rearrange, HloInstruction* operand);
  bool IsReshapeMoveCandidate(HloInstruction* instruction);
  const HloInstruction* FirstNontrivialRearrange(
      absl::Span<const HloInstruction* const> instrs);
  bool CanTriviallyRearrange(const HloInstruction* instr,
                             const HloInstruction* rearrange);

  ReshapeMoverOptions options_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_
