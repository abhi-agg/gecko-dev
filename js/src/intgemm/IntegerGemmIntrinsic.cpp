/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "intgemm/IntegerGemmIntrinsic.h"
#include "mozilla/CheckedInt.h"
#include <intgemm.h>

#include <utility>

#include "vm/JSContext.h"
#include "wasm/WasmInstance.h"
#include "wasm/WasmLog.h"
#include "vm/ArrayBufferObject-inl.h"

namespace js {
namespace intgemm {

static constexpr uint8_t ARRAY_ALIGNMENT = 64;
static constexpr uint8_t ROWS_A_MULTIPLIER = 1;
static constexpr uint8_t COLUMNS_A_MULTIPLIER = 64;
static constexpr uint8_t ROWS_B_MULTIPLIER = COLUMNS_A_MULTIPLIER;
static constexpr uint8_t COLUMNS_B_MULTIPLIER = 8;
static constexpr uint8_t SELECTED_COLUMNS_B_MULTIPLIER = 8;

void ReportError(JSContext* cx, const unsigned errorNumber) {
  JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr, errorNumber);
}

size_t GetWasmRawBufferLength(const uint8_t* memBase) {
  const js::WasmArrayRawBuffer* rawBuf =
      js::WasmArrayRawBuffer::fromDataPtr(memBase);
  return rawBuf->byteLength();
}

bool IsOverflowCheckPassed(uint64_t value) {
  mozilla::CheckedUint64 valueCheckedUint64(value);
  return valueCheckedUint64.isValid();
}

bool IsMemoryBoundCheckPassed(uint32_t input, uint64_t inputSize,
                              const size_t wasmBufferLimit) {
  uint64_t inputUpperLimit = (uint64_t)input + inputSize;
  return IsOverflowCheckPassed(inputUpperLimit) &&
         (inputUpperLimit < wasmBufferLimit);
}

bool IsAlignmentCheckPassed(const uint8_t* ptr) {
  return ((reinterpret_cast<uintptr_t>(ptr) % ARRAY_ALIGNMENT) == 0);
}

}  // namespace intgemm
}  // namespace js

int32_t js::intgemm::IntrI8PrepareB(wasm::Instance* instance,
                                    uint32_t inputMatrixB, float scale,
                                    float zeroPoint, uint32_t rowsB,
                                    uint32_t colsB, uint32_t outputMatrixB,
                                    uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8PrepareB.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  matrixSize:%" PRIu64,
              __FUNCTION__, rowsB, colsB, matrixSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixB, matrixSize, wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixB:%x  scale:%f  "
              "zeroPoint:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  outputMatrixB:%x  matrixSize:%" PRIu64 "  wasmBufferLimit:%zu",
              __FUNCTION__, inputMatrixB, scale, zeroPoint, rowsB, colsB,
              outputMatrixB, matrixSize, wasmBufferLimit);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPtr = &memBase[inputMatrixB];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixBPtr) ||
      !IsAlignmentCheckPassed(outputMatrixBPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Pointer alignment checks failed. inputMatrixB:%x  scale:%f  "
              "zeroPoint:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  outputMatrixB:%x  inputMatrixBPtr:%p  outputMatrixBPtr:%p",
              __FUNCTION__, inputMatrixB, scale, zeroPoint, rowsB, colsB,
              outputMatrixB, inputMatrixBPtr, outputMatrixBPtr);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm) for PrepareB
  ::intgemm::Int8::PrepareB((const float*)inputMatrixBPtr,
                            (int8_t*)outputMatrixBPtr,
                            (float)scale,  // Quant Mult
                            rowsB, colsB);
  return 0;
}

int32_t js::intgemm::IntrI8PrepareBFromTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBTransposed, float scale,
    float zeroPoint, uint32_t rowsB, uint32_t colsB, uint32_t outputMatrixB,
    uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8PrepareBFromTransposed.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  matrixSize:%" PRIu64,
              __FUNCTION__, rowsB, colsB, matrixSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixBTransposed, matrixSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixBTransposed:%x  "
              "scale:%f  zeroPoint:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  outputMatrixB:%x  matrixSize:%" PRIu64 "  wasmBufferLimit:%zu",
              __FUNCTION__, inputMatrixBTransposed, scale, zeroPoint, rowsB,
              colsB, outputMatrixB, matrixSize, wasmBufferLimit);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBTransposedPtr = &memBase[inputMatrixBTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixBTransposedPtr) ||
      !IsAlignmentCheckPassed(outputMatrixBPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(
        cx,
        "%s: Pointer alignment checks failed. inputMatrixBTransposed:%x  "
        "scale:%f  zeroPoint:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
        "  outputMatrixB:%x  inputMatrixBTransposedPtr:%p  outputMatrixBPtr:%p",
        __FUNCTION__, inputMatrixBTransposed, scale, zeroPoint, rowsB, colsB,
        outputMatrixB, inputMatrixBTransposedPtr, outputMatrixBPtr);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm) for PrepareBTransposed
  ::intgemm::Int8::PrepareBTransposed((const float*)inputMatrixBTransposedPtr,
                                      (int8_t*)outputMatrixBPtr,
                                      (float)scale,  // Quant Mult
                                      rowsB, colsB);
  return 0;
}

int32_t js::intgemm::IntrI8PrepareBFromQuantizedTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBQuantizedTransposed,
    uint32_t rowsB, uint32_t colsB, uint32_t outputMatrixB, uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8PrepareBFromQuantizedTransposed.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  matrixSize:%" PRIu64,
              __FUNCTION__, rowsB, colsB, matrixSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixBQuantizedTransposed, matrixSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. "
              "inputMatrixBQuantizedTransposed:%x  rowsB:%" PRIu32
              "  colsB:%" PRIu32 "  outputMatrixB:%x  matrixSize:%" PRIu64
              "  wasmBufferLimit:%zu",
              __FUNCTION__, inputMatrixBQuantizedTransposed, rowsB, colsB,
              outputMatrixB, matrixSize, wasmBufferLimit);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBQuantizedTransposedPtr =
      &memBase[inputMatrixBQuantizedTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixBQuantizedTransposedPtr) ||
      !IsAlignmentCheckPassed(outputMatrixBPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(
        cx,
        "%s: Pointer alignment checks failed. "
        "inputMatrixBQuantizedTransposed:%x  rowsB:%" PRIu32 "  colsB:%" PRIu32
        "  outputMatrixB:%x  inputMatrixBQuantizedTransposedPtr:%p  "
        "outputMatrixBPtr:%p",
        __FUNCTION__, inputMatrixBQuantizedTransposed, rowsB, colsB,
        outputMatrixB, inputMatrixBQuantizedTransposedPtr, outputMatrixBPtr);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8::PrepareBQuantizedTransposed(
      (const int8_t*)inputMatrixBQuantizedTransposedPtr,
      (int8_t*)outputMatrixBPtr, rowsB, colsB);
  return 0;
}

int32_t js::intgemm::IntrI8PrepareA(wasm::Instance* instance,
                                    uint32_t inputMatrixA, float scale,
                                    float zeroPoint, uint32_t rowsA,
                                    uint32_t colsA, uint32_t outputMatrixA,
                                    uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8PrepareA.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsA * (uint64_t)colsA;
  if ((matrixSize == 0) || (rowsA % ROWS_A_MULTIPLIER != 0) ||
      (colsA % COLUMNS_A_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsA:%" PRIu32 "  colsA:%" PRIu32
              "  matrixSize:%" PRIu64,
              __FUNCTION__, rowsA, colsA, matrixSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixA, matrixSize, wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(outputMatrixA, matrixSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixA:%x  scale:%f  "
              "zeroPoint:%f  rowsA:%" PRIu32 "  colsA:%" PRIu32
              "  outputMatrixA:%x  matrixSize:%" PRIu64 "  wasmBufferLimit:%zu",
              __FUNCTION__, inputMatrixA, scale, zeroPoint, rowsA, colsA,
              outputMatrixA, matrixSize, wasmBufferLimit);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPtr = &memBase[inputMatrixA];
  uint8_t* outputMatrixAPtr = &memBase[outputMatrixA];

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixAPtr) ||
      !IsAlignmentCheckPassed(outputMatrixAPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Pointer alignment checks failed. inputMatrixA:%x  scale:%f  "
              "zeroPoint:%f  rowsA:%" PRIu32 "  colsA:%" PRIu32
              "  outputMatrixA:%x  inputMatrixAPtr:%p  outputMatrixAPtr:%p",
              __FUNCTION__, inputMatrixA, scale, zeroPoint, rowsA, colsA,
              outputMatrixA, inputMatrixAPtr, outputMatrixAPtr);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8Shift::PrepareA((const float*)inputMatrixAPtr,
                                 (int8_t*)outputMatrixAPtr, scale, rowsA,
                                 colsA);
  return 0;
}

int32_t js::intgemm::IntrI8PrepareBias(
    wasm::Instance* instance, uint32_t inputMatrixBPrepared, float scaleA,
    float zeroPointA, float scaleB, float zeroPointB, uint32_t rowsB,
    uint32_t colsB, uint32_t inputBias, uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8PrepareBias.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  matrixSize:%" PRIu64,
              __FUNCTION__, rowsB, colsB, matrixSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(inputBias, colsB, wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(output, colsB, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixBPrepared:%x  "
              "scaleA:%f  zeroPointA:%f  scaleB:%f  "
              "zeroPointB:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  inputBias:%x  outputBias:%x  matrixSize:%" PRIu64
              "  wasmBufferLimit:%zu",
              __FUNCTION__, inputMatrixBPrepared, scaleA, zeroPointA, scaleB,
              zeroPointB, rowsB, colsB, inputBias, output, matrixSize,
              wasmBufferLimit);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPtr = &memBase[inputBias];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor =
      (-1) * ((127.0f / scaleA) * (127.0f / scaleB)) / (127.0f);

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixBPrepared:%x  "
              "scaleA:%f  zeroPointA:%f  scaleB:%f  "
              "zeroPointB:%f  rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  inputBias:%x  outputBias:%x  inputMatrixBPreparedPtr:%p  "
              "inputBiasPtr:%p  outputBiasPtr:%p  unquantFactor:%f",
              __FUNCTION__, inputMatrixBPrepared, scaleA, zeroPointA, scaleB,
              zeroPointB, rowsB, colsB, inputBias, output,
              inputMatrixBPreparedPtr, inputBiasPtr, outputPtr, unquantFactor);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8Shift::PrepareBias(
      (const int8_t*)inputMatrixBPreparedPtr, rowsB, colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPtr, (float*)outputPtr));
  return 0;
}

int32_t js::intgemm::IntrI8MultiplyAndAddBias(
    wasm::Instance* instance, uint32_t inputMatrixAPrepared, float scaleA,
    float zeroPointA, uint32_t inputMatrixBPrepared, float scaleB,
    float zeroPointB, uint32_t inputBiasPrepared, float unquantMultiplier,
    uint32_t rowsA, uint32_t width, uint32_t colsB, uint32_t output,
    uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8MultiplyAndAddBias.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixASize = (uint64_t)rowsA * (uint64_t)width;
  uint64_t matrixBSize = (uint64_t)width * (uint64_t)colsB;
  uint64_t inputBiasSize = (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsA * (uint64_t)colsB;
  if ((matrixASize == 0) || (matrixBSize == 0) ||
      (rowsA % ROWS_A_MULTIPLIER != 0) || (width % COLUMNS_A_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsA:%" PRIu32 "  width:%" PRIu32
              "  colsB:%" PRIu32 "  matrixASize:%" PRIu64
              "  matrixBSize:%" PRIu64,
              __FUNCTION__, rowsA, width, colsB, matrixASize, matrixBSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixAPrepared, matrixASize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(inputMatrixBPrepared, matrixBSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(inputBiasPrepared, inputBiasSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(output, outputSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixAPrepared:%x  "
              "scaleA:%f  zeroPointA:%f  "
              "inputMatrixBPrepared:%x  scaleB:%f  zeroPointB:%f  "
              "inputBiasPrepared:%x "
              " unquantMultiplier:%f  rowsA:%" PRIu32 "  width:%" PRIu32
              "  colsB:%" PRIu32 "  output:%x  matrixASize:%" PRIu64
              "  matrixBSize:%" PRIu64 "  inputBiasSize:%" PRIu64
              "  outputSize:%" PRIu64,
              __FUNCTION__, inputMatrixAPrepared, scaleA, zeroPointA,
              inputMatrixBPrepared, scaleB, zeroPointB, inputBiasPrepared,
              unquantMultiplier, rowsA, width, colsB, output, matrixASize,
              matrixBSize, inputBiasSize, outputSize);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPreparedPtr = &memBase[inputMatrixAPrepared];
  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPreparedPtr = &memBase[inputBiasPrepared];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor = unquantMultiplier / (scaleA * scaleB);

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixAPreparedPtr) ||
      !IsAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(
        cx,
        "%s: Pointer alignment checks failed. inputMatrixAPrepared:%x  "
        "scaleA:%f  zeroPointA:%f  "
        "inputMatrixBPrepared:%x  scaleB:%f  zeroPointB:%f  "
        "inputBiasPrepared:%x "
        " unquantMultiplier:%f  rowsA:%" PRIu32 "  width:%" PRIu32
        "  colsB:%" PRIu32
        "  output:%x  inputMatrixAPreparedPtr:%p  inputMatrixBPreparedPtr:%p  "
        "inputBiasPreparedPtr:%p  outputPtr:%p  unquantFactor:%f",
        __FUNCTION__, inputMatrixAPrepared, scaleA, zeroPointA,
        inputMatrixBPrepared, scaleB, zeroPointB, inputBiasPrepared,
        unquantMultiplier, rowsA, width, colsB, output, inputMatrixAPreparedPtr,
        inputMatrixBPreparedPtr, inputBiasPreparedPtr, outputPtr,
        unquantFactor);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8Shift::Multiply(
      (const int8_t*)inputMatrixAPreparedPtr,
      (const int8_t*)inputMatrixBPreparedPtr, rowsA, width, colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPreparedPtr,
          (float*)outputPtr));
  return 0;
}

int32_t js::intgemm::IntrI8SelectColumnsOfB(wasm::Instance* instance,
                                            uint32_t inputMatrixBPrepared,
                                            uint32_t rowsB, uint32_t colsB,
                                            uint32_t colIndexList,
                                            uint32_t sizeColIndexList,
                                            uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(wasm::SASigIntrI8SelectColumnsOfB.failureMode ==
             wasm::FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsB * (uint64_t)sizeColIndexList;
  if ((matrixSize == 0) || (outputSize == 0) ||
      (rowsB % ROWS_B_MULTIPLIER != 0) || (colsB % COLUMNS_B_MULTIPLIER != 0) ||
      (sizeColIndexList % SELECTED_COLUMNS_B_MULTIPLIER != 0)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Matrix size checks failed. rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  matrixSize:%" PRIu64 "  outputSize:%" PRIu64,
              __FUNCTION__, rowsB, colsB, matrixSize, outputSize);
    ReportError(cx, JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  size_t wasmBufferLimit = GetWasmRawBufferLength(memBase);
  if (!IsMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(colIndexList, sizeColIndexList,
                                wasmBufferLimit) ||
      !IsMemoryBoundCheckPassed(output, outputSize, wasmBufferLimit)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Memory Bound checks failed. inputMatrixBPrepared:%x  "
              "rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  colIndexList:%x  sizeColIndexList:%" PRIu32
              " output:%x  matrixSize:%" PRIu64 "  outputSize:%" PRIu64,
              __FUNCTION__, inputMatrixBPrepared, rowsB, colsB, colIndexList,
              sizeColIndexList, output, matrixSize, outputSize);
    ReportError(cx, JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* colIndexListPtr = &memBase[colIndexList];
  uint8_t* outputPtr = &memBase[output];

  // Pointer Alignment checks for matricies
  if (!IsAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    JSContext* cx = TlsContext.get();
    wasm::Log(cx,
              "%s: Pointer alignment checks failed. inputMatrixBPrepared:%x  "
              "rowsB:%" PRIu32 "  colsB:%" PRIu32
              "  colIndexList:%x  sizeColIndexList:%" PRIu32
              " output:%x  inputMatrixBPreparedPtr:%p  colIndexListPtr:%p  "
              "outputPtr:%p",
              __FUNCTION__, inputMatrixBPrepared, rowsB, colsB, colIndexList,
              sizeColIndexList, output, inputMatrixBPreparedPtr,
              colIndexListPtr, outputPtr);
    ReportError(cx, JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8::SelectColumnsB(
      (const int8_t*)inputMatrixBPreparedPtr, (int8_t*)outputPtr, rowsB,
      (const uint32_t*)colIndexListPtr,
      (const uint32_t*)colIndexListPtr + sizeColIndexList);
  return 0;
}
