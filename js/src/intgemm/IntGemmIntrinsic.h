/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * Copyright 2021 Mozilla Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef intgemm_intrinsic_h
#define intgemm_intrinsic_h

#include <stdint.h>

namespace js {
namespace wasm {
  class Instance;
}

namespace intgemm {

  int32_t intrSample1(wasm::Instance* instance, uint32_t arr,
                                          uint32_t len, uint8_t* memBase);

  // i8PrepareB(input_matrix_B: i32, scale: f32, zero_point: f32, rows_B: i32, cols_B: i32, output_matrix_B: i32)
  int32_t intrI8PrepareB(wasm::Instance* instance,
             uint32_t input_matrix_B,
             float scale,
             float zero_point,
             uint32_t rows_B,
             uint32_t cols_B,
             uint32_t output_matrix_B,
             uint8_t* memBase);
}
}

#endif // intgemm_intrinsic
