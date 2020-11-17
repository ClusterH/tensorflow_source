/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_time.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT

namespace tflite {
namespace {

constexpr int kTimerNum = 1;            // Use CTIMER 1 as benchmarking timer - 
                                        // this timer must not be used elsewhere.
constexpr int kClocksPerSecond = 12e6;  // Clock set to operate at 12MHz.

}  // namespace

int32_t ticks_per_second() { return kClocksPerSecond; }

// Calling this method enables a timer that runs for eternity. The user is
// responsible for avoiding trampling on this timer's config, otherwise timing
// measurements may no longer be valid.
int32_t GetCurrentTimeTicks() {
  // TODO(b/150808076): Split out initialization, intialize in interpreter.
  static bool is_initialized = false;
  if (!is_initialized) {
    am_hal_ctimer_config_t timer_config;
    // Operate as a 32-bit timer.
    timer_config.ui32Link = 1;
    // Set timer A to continuous mode at 12MHz.
    timer_config.ui32TimerAConfig =
        AM_HAL_CTIMER_FN_CONTINUOUS | AM_HAL_CTIMER_HFRC_12MHZ;

    am_hal_ctimer_stop(kTimerNum, AM_HAL_CTIMER_BOTH);
    am_hal_ctimer_clear(kTimerNum, AM_HAL_CTIMER_BOTH);
    am_hal_ctimer_config(kTimerNum, &timer_config);
    am_hal_ctimer_start(kTimerNum, AM_HAL_CTIMER_TIMERA);
    is_initialized = true;
  }
  return CTIMERn(kTimerNum)->TMR0;
}

}  // namespace tflite
