/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "intgemm/IntGemmIntrinsic.h"
#include <intgemm.h>

#include <utility>

#include "vm/JSContext.h"
#include "wasm/WasmInstance.h"

#include "vm/ArrayBufferObject-inl.h"

using namespace js::wasm;

#define INTGEMM_INTR_SHARED 0

int32_t js::intgemm::intrSample1(Instance* instance, uint32_t arr, uint32_t len,
                                 uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrSample1.failureMode == FailureMode::FailOnNegI32);

#if INTGEMM_INTR_SHARED
  const SharedArrayRawBuffer* rawBuf =
      SharedArrayRawBuffer::fromDataPtr(memBase);
  size_t memLen = rawBuf->volatileByteLength();
  // TODO shall be more carefull with using shared buffer
#else
  const WasmArrayRawBuffer* rawBuf = WasmArrayRawBuffer::fromDataPtr(memBase);
  size_t memLen = rawBuf->byteLength();
#endif
  // Bounds check and deal with arithmetic overflow.
  uint64_t destLimit = uint64_t(arr) + uint64_t(len);
  if (destLimit > memLen) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  if (len == 0) return 0;
  uint8_t* destPtr = &memBase[arr];
  for (uint32_t i = 0, j = len - 1; i < j; i++, j--) {
    std::swap(destPtr[i], destPtr[j]);
  }

  return 0;
}

int32_t js::intgemm::intrI8PrepareB(wasm::Instance* instance,
                                    uint32_t inputMatrixB, float scale,
                                    float zeroPoint, uint32_t rowsB,
                                    uint32_t colsB, uint32_t outputMatrixB,
                                    uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

#if INTGEMM_INTR_SHARED
  const SharedArrayRawBuffer* rawBuf =
      SharedArrayRawBuffer::fromDataPtr(memBase);
  size_t memLen = rawBuf->volatileByteLength();
  // TODO shall be more carefull with using shared buffer
#else
  const WasmArrayRawBuffer* rawBuf = WasmArrayRawBuffer::fromDataPtr(memBase);
  size_t memLen = rawBuf->byteLength();
#endif

  // Size of matrix shouldn't be zero
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (matrixSize == 0) {
    JSContext* cx = TlsContext.get();
    // ToDo: Some meaningful error message?
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  // Bounds check and deal with arithmetic overflow for input matrix
  // ToDo: Should there be a bound check for inputMatrixB?
  uint64_t inputDestLimit = uint64_t(inputMatrixB) + matrixSize;
  if (inputDestLimit > memLen) {
    JSContext* cx = TlsContext.get();
    // ToDo: Some meaningful error message?
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  // Bounds check and deal with arithmetic overflow for output matrix
  // ToDo: Should there be a bound check for outputMatrixB?
  uint64_t outputDestLimit = uint64_t(outputMatrixB) + matrixSize;
  if (outputDestLimit > memLen) {
    JSContext* cx = TlsContext.get();
    // ToDo: Some meaningful error message?
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm) for PrepareB
  ::intgemm::Int8::PrepareB((const float*)inputMatrixB, (int8_t*)outputMatrixB,
                            (float)scale,  // Quant Mult
                            (Index)rowsB, (Index)colsB);
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBTransposed, float scale,
    float zeroPoint, Index rowsB, Index colsB, uint32_t outputMatrixB,
    uint8_t* memBase) {
  // JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
  //                             JSMSG_WASM_OUT_OF_BOUNDS);
  return -1;
}

int32_t js::intgemm::intrI8PrepareBFromQuantizedTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBQuantizedTransposed,
    Index rowsB, Index colsB, uint32_t outputMatrixB, uint8_t* memBase) {
  // ToDo: Write implementation
  ::intgemm::Int8::PrepareBQuantizedTransposed(
      (const int8_t*)inputMatrixBQuantizedTransposed, (int8_t*)outputMatrixB,
      (Index)rowsB, (Index)colsB);
  return 0;
}

int32_t js::intgemm::intrI8PrepareA(wasm::Instance* instance,
                                    uint32_t inputMatrixA, float scale,
                                    float zeroPoint, Index rowsA, Index colsA,
                                    uint32_t outputMatrixA, uint8_t* memBase) {
  // ToDo: Write implementation
  ::intgemm::Int8Shift::PrepareA((const float*)inputMatrixA,
                                 (int8_t*)outputMatrixA, scale, (Index)rowsA,
                                 (Index)colsA);
  return 0;
}

int32_t js::intgemm::intrI8PrepareBias(
    wasm::Instance* instance, uint32_t inputMatrixBPrepared, float scaleA,
    float zeroPointA, float scaleB, float zeroPointB, Index rowsB, Index colsB,
    uint32_t inputBias, uint32_t output, uint8_t* memBase) {
  // ToDo: Write implementation
  float unquantFactor =
      (-1) * ((127.0f / scaleA) * (127.0f / scaleB)) / (127.0f);
  ::intgemm::Int8Shift::PrepareBias(
      (const int8_t*)inputMatrixBPrepared, (Index)rowsB, (Index)colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBias, (float*)output));
  return 0;
}

int32_t js::intgemm::intrI8MultiplyAndAddBias(
    wasm::Instance* instance, uint32_t inputMatrixAPrepared, float scaleA,
    float zeroPointA, uint32_t inputMatrixBPrepared, float scaleB,
    float zeroPointB, uint32_t inputBiasPrepared, float unquantMultiplier,
    Index rowsA, Index width, Index colsB, uint32_t output, uint8_t* memBase) {
  // ToDo: Write implementation
  float unquantFactor = unquantMultiplier / (scaleA * scaleB);
  ::intgemm::Int8Shift::Multiply(
      (const int8_t*)inputMatrixAPrepared, (const int8_t*)inputMatrixBPrepared,
      (Index)rowsA, (Index)width, (Index)colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPrepared, (float*)output));
  return 0;
}

int32_t js::intgemm::intrI8SelectColumnsOfB(wasm::Instance* instance,
                               uint32_t inputMatrixBPrepared, Index rowsB,
                               Index colsB, Index colIndexList,
                               Index sizeColIndexList, uint32_t output,
                               uint8_t* memBase) {
  // ToDo: Write implementation
  ::intgemm::Int8::SelectColumnsB((const int8_t*)inputMatrixBPrepared,
                                  (int8_t*)output, (Index)rowsB, colIndexList,
                                  colIndexList + sizeColIndexList);
  return 0;
}
