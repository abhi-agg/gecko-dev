/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef intgemm_IntGemmIntrinsic_h
#define intgemm_IntGemmIntrinsic_h

#include <stdint.h>

namespace js {
namespace wasm {
class Instance;
}

namespace intgemm {

/**
 * The definition of all the functions that implement integer gemm intrinsics.
 * Please refer to $TOPSRCDIR/js/src/wasm/WasmIntrinsic.yaml for details.
 */

// int8_prepare_b(inputMatrixB: i32, scale: f32, zeroPoint: f32, rowsB: i32,
// colsB: i32, outputMatrixB: i32).
int32_t intrI8PrepareB(wasm::Instance* instance, uint32_t inputMatrixB,
                       float scale, float zeroPoint, uint32_t rowsB,
                       uint32_t colsB, uint32_t outputMatrixB,
                       uint8_t* memBase);

// int8_prepare_b_from_transposed(inputMatrixBTransposed: i32, scale: f32,
// zeroPoint: f32, rowsB: i32, colsB: i32, outputMatrixB: i32)
int32_t intrI8PrepareBFromTransposed(wasm::Instance* instance,
                                     uint32_t inputMatrixBTransposed,
                                     float scale, float zeroPoint,
                                     uint32_t rowsB, uint32_t colsB,
                                     uint32_t outputMatrixB, uint8_t* memBase);

// int8_prepare_b_from_quantized_transposed(inputMatrixBQuantizedTransposed:
// i32, rowsB: i32, colsB: i32, outputMatrixB: i32)
int32_t intrI8PrepareBFromQuantizedTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBQuantizedTransposed,
    uint32_t rowsB, uint32_t colsB, uint32_t outputMatrixB, uint8_t* memBase);

// int8_prepare_a(inputMatrixA: i32, scale: f32, zeroPoint: f32, rowsA: i32,
// colsA: i32, outputMatrixA: i32)
int32_t intrI8PrepareA(wasm::Instance* instance, uint32_t inputMatrixA,
                       float scale, float zeroPoint, uint32_t rowsA,
                       uint32_t colsA, uint32_t outputMatrixA,
                       uint8_t* memBase);

// int8_prepare_bias(inputMatrixBPrepared: i32, scaleA: f32, zeroPointA: f32,
// scaleB: f32, zeroPointB: f32, rowsB: i32, colsB: i32, inputBias: i32, output:
// i32)
int32_t intrI8PrepareBias(wasm::Instance* instance,
                          uint32_t inputMatrixBPrepared, float scaleA,
                          float zeroPointA, float scaleB, float zeroPointB,
                          uint32_t rowsB, uint32_t colsB, uint32_t inputBias,
                          uint32_t output, uint8_t* memBase);

// int8_multiply_and_add_bias(inputMatrixAPrepared: i32, scaleA: f32,
// zeroPointA: f32, inputMatrixBPrepared: i32, scaleB: f32, zeroPointB: f32,
// inputBiasPrepared: i32, unquantMultiplier: f32, rowsA: i32, width: i32,
// colsB: i32, output: i32)
#if 1
int32_t intrI8MultiplyAndAddBias(wasm::Instance* instance,
                                 uint32_t inputMatrixAPrepared, float scaleA,
                                 float zeroPointA,
                                 uint32_t inputMatrixBPrepared, float scaleB,
                                 float zeroPointB, uint32_t inputBiasPrepared,
                                 float unquantMultiplier, uint32_t rowsA,
                                 uint32_t width, uint32_t colsB,
                                 uint32_t output, uint8_t* memBase);
#else
// The same intrinsic function with reduced no. of arguments (using structure
// approach)
int32_t intrI8MultiplyAndAddBias(wasm::Instance* instance,
                                 uint32_t inputMatrixAPrepared,
                                 uint32_t inputMatrixBPrepared,
                                 uint32_t inputBiasPrepared,
                                 float unquantMultiplier, uint32_t rowsA,
                                 uint32_t width, uint32_t colsB,
                                 uint32_t output, uint8_t* memBase);
#endif

// int8_select_columns_of_b(inputMatrixBPrepared: i32, rowsB: i32, colsB: i32,
// colIndexList: i32, sizeColIndexList: i32, output: i32)
int32_t intrI8SelectColumnsOfB(wasm::Instance* instance,
                               uint32_t inputMatrixBPrepared, uint32_t rowsB,
                               uint32_t colsB, uint32_t colIndexList,
                               uint32_t sizeColIndexList, uint32_t output,
                               uint8_t* memBase);

}  // namespace intgemm
}  // namespace js

#endif  // intgemm_IntGemmIntrinsic_h
