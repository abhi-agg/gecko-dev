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
  // The data type of rows and cols are dependent on intgemm library.
  // Please check $TOPSRCDIR/third_party/intgemm/intgemm/intgemm.h
  using Index = uint32_t;

  int32_t intrSample1(wasm::Instance* instance, uint32_t arr,
                                          uint32_t len, uint8_t* memBase);

  // i8PrepareB(inputMatrixB: i32, scale: f32, zeroPoint: f32, rowsB: i32, colsB: i32, outputMatrixB: i32)
  int32_t intrI8PrepareB(wasm::Instance* instance,
             uint32_t inputMatrixB,
             float scale,
             float zeroPoint,
             Index rowsB,
             Index colsB,
             uint32_t outputMatrixB,
             uint8_t* memBase);

  // i8PrepareBFromTransposed(inputMatrixBTransposed: i32, scale: f32, zeroPoint: f32, rowsB: i32, colsB: i32, outputMatrixB: i32)
  int32_t intrI8PrepareBFromTransposed(wasm::Instance* instance,
             uint32_t inputMatrixBTransposed,
             float scale,
             float zeroPoint,
             Index rowsB,
             Index colsB,
             uint32_t outputMatrixB,
             uint8_t* memBase);

  // i8PrepareBFromQuantizedTransposed(inputMatrixBQuantizedTransposed: i32, rowsB: i32, colsB: i32, outputMatrixB: i32)
  int32_t intrI8PrepareBFromQuantizedTransposed(wasm::Instance* instance,
             uint32_t inputMatrixBQuantizedTransposed,
             Index rowsB,
             Index colsB,
             uint32_t outputMatrixB,
             uint8_t* memBase);

  // i8PrepareA(inputMatrixA: i32, scale: f32, zeroPoint: f32, rowsA: i32, colsA: i32, outputMatrixA: i32)
  int32_t intrI8PrepareA(wasm::Instance* instance,
             uint32_t inputMatrixA,
             float scale,
             float zeroPoint,
             Index rowsA,
             Index colsA,
             uint32_t outputMatrixA,
             uint8_t* memBase);
}
}

#endif // intgemm_intrinsic
