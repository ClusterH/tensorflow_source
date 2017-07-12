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
// GENERATED FILE. To update, edit tftypes.pl instead.

package org.tensorflow.types;

/** Represents a 32-bit signed integer. */
public class TFInt32 implements TFType {
  private TFInt32() {}
  static {
    Types.typeCodes.put(TFInt32.class, org.tensorflow.DataType.INT32);
  }
  static {
    Types.scalars.put(TFInt32.class, 0);
  }
}
