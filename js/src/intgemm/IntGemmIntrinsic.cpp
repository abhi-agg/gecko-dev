/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "intgemm/IntGemmIntrinsic.h"
#include "mozilla/Logging.h"
#include <intgemm.h>

#include <utility>

#include "vm/JSContext.h"
#include "wasm/WasmInstance.h"

#include "vm/ArrayBufferObject-inl.h"

using namespace js::wasm;

namespace mozilla {
  static mozilla::LazyLogModule gIntGemmLog("IntGemmLog");
  #define LOG(level, ...) MOZ_LOG(gIntGemmLog, mozilla::LogLevel::level, (__VA_ARGS__))
}

namespace js {
namespace intgemm {

#define INTGEMM_INTR_SHARED 0

size_t getWasmRawBufferLength(uint8_t* memBase) {
  #if INTGEMM_INTR_SHARED
    const js::SharedArrayRawBuffer* rawBuf =
        js::SharedArrayRawBuffer::fromDataPtr(memBase);
    return rawBuf->volatileByteLength();
    // TODO shall be more carefull with using shared buffer
  #else
    const js::WasmArrayRawBuffer* rawBuf = js::WasmArrayRawBuffer::fromDataPtr(memBase);
    return rawBuf->byteLength();
  #endif
}

bool isMemoryBoundCheckPassed(uint32_t input, uint64_t inputSize, size_t wasmBufferLimit) {
  //ToDo: Deal with arithmetic overflow
  uint64_t inputUpperLimit = (uint64_t)input + inputSize;
  return (inputUpperLimit > wasmBufferLimit) ? false : true;
}
}
}

int32_t js::intgemm::intrSample1(Instance* instance, uint32_t arr, uint32_t len,
                                 uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrSample1.failureMode == FailureMode::FailOnNegI32);

  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  if (!isMemoryBoundCheckPassed(arr, len, wasmBufferLen)) {
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
  fprintf(stderr, "intrI8PrepareB called with inputMatrixB:%d outputMatrixB:%d\n", inputMatrixB, outputMatrixB);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixB, matrixSize, wasmBufferLen) || !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPtr = &memBase[inputMatrixB];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];
  // Actual call to the 3rd party library (intgemm) for PrepareB
  fprintf(stderr, "Calling Int8::PrepareB\n");
  ::intgemm::Int8::PrepareB((const float*)inputMatrixBPtr,
                            (int8_t*)outputMatrixBPtr,
                            (float)scale,  // Quant Mult
                            (Index)rowsB, (Index)colsB);
  fprintf(stderr, "Done Int8::PrepareB\n");
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
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixBQuantizedTransposed, matrixSize, wasmBufferLen) || !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBQuantizedTransposedPtr =
      &memBase[inputMatrixBQuantizedTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];
  fprintf(stderr, "Calling Int8::PrepareBQuantizedTransposed\n");
  ::intgemm::Int8::PrepareBQuantizedTransposed(
      (const int8_t*)inputMatrixBQuantizedTransposedPtr,
      (int8_t*)outputMatrixBPtr, (Index)rowsB, (Index)colsB);
  fprintf(stderr, "Done Int8::PrepareBQuantizedTransposed\n");
  return 0;
}

int32_t js::intgemm::intrI8PrepareA(wasm::Instance* instance,
                                    uint32_t inputMatrixA, float scale,
                                    float zeroPoint, Index rowsA, Index colsA,
                                    uint32_t outputMatrixA, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsA * (uint64_t)colsA;
  if (!isMemoryBoundCheckPassed(inputMatrixA, matrixSize, wasmBufferLen) || !isMemoryBoundCheckPassed(outputMatrixA, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPtr = &memBase[inputMatrixA];
  uint8_t* outputMatrixAPtr = &memBase[outputMatrixA];
  fprintf(stderr, "Calling Int8Shift::PrepareA\n");
  ::intgemm::Int8Shift::PrepareA((const float*)inputMatrixAPtr,
                                 (int8_t*)outputMatrixAPtr, scale, (Index)rowsA,
                                 (Index)colsA);
  fprintf(stderr, "Done Int8Shift::PrepareA\n");
  return 0;
}

int32_t js::intgemm::intrI8PrepareBias(
    wasm::Instance* instance, uint32_t inputMatrixBPrepared, float scaleA,
    float zeroPointA, float scaleB, float zeroPointB, Index rowsB, Index colsB,
    uint32_t inputBias, uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize, wasmBufferLen) || !isMemoryBoundCheckPassed(inputBias, colsB, wasmBufferLen) || !isMemoryBoundCheckPassed(output, colsB, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPtr = &memBase[inputBias];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor =
      (-1) * ((127.0f / scaleA) * (127.0f / scaleB)) / (127.0f);
  fprintf(stderr, "Calling Int8Shift::PrepareBias\n");
  ::intgemm::Int8Shift::PrepareBias(
      (const int8_t*)inputMatrixBPreparedPtr, (Index)rowsB, (Index)colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPtr, (float*)outputPtr));
  fprintf(stderr, "Done Int8Shift::PrepareBias\n");
  return 0;
}

int32_t js::intgemm::intrI8MultiplyAndAddBias(
    wasm::Instance* instance, uint32_t inputMatrixAPrepared, float scaleA,
    float zeroPointA, uint32_t inputMatrixBPrepared, float scaleB,
    float zeroPointB, uint32_t inputBiasPrepared, float unquantMultiplier,
    Index rowsA, Index width, Index colsB, uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixASize = (uint64_t)rowsA * (uint64_t)width;
  uint64_t matrixBSize = (uint64_t)width * (uint64_t)colsB;
  uint64_t inputBiasSize = (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsA * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixAPrepared, matrixASize, wasmBufferLen) || !isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixBSize, wasmBufferLen) || !isMemoryBoundCheckPassed(inputBiasPrepared, inputBiasSize, wasmBufferLen) || !isMemoryBoundCheckPassed(output, outputSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  // ToDo: Write implementation
  uint8_t* inputMatrixAPreparedPtr = &memBase[inputMatrixAPrepared];
  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPreparedPtr = &memBase[inputBiasPrepared];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor = unquantMultiplier / (scaleA * scaleB);
  fprintf(stderr, "Calling Int8Shift::Multiply: A:%p, B:%p, Bias:%p, output:%p, unquantFactor:%f, rows:%d, width:%d, cols:%d\n", inputMatrixAPreparedPtr, inputMatrixBPreparedPtr, inputBiasPreparedPtr, outputPtr, unquantFactor, rowsA, width, colsB);
  ::intgemm::Int8Shift::Multiply(
      (const int8_t*)inputMatrixAPreparedPtr,
      (const int8_t*)inputMatrixBPreparedPtr, (Index)rowsA, (Index)width,
      (Index)colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPreparedPtr,
          (float*)outputPtr));
  fprintf(stderr, "Done Int8Shift::Multiply\n");
  return 0;
}

int32_t js::intgemm::intrI8SelectColumnsOfB(wasm::Instance* instance,
                                            uint32_t inputMatrixBPrepared,
                                            Index rowsB, Index colsB,
                                            Index colIndexList,
                                            Index sizeColIndexList,
                                            uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsB * (uint64_t)sizeColIndexList;
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize, wasmBufferLen) || !isMemoryBoundCheckPassed(colIndexList, sizeColIndexList, wasmBufferLen) || !isMemoryBoundCheckPassed(output, outputSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* colIndexListPtr = &memBase[colIndexList];
  uint8_t* outputPtr = &memBase[output];
  fprintf(stderr, "Calling Int8::SelectColumnsB\n");
  ::intgemm::Int8::SelectColumnsB(
      (const int8_t*)inputMatrixBPreparedPtr, (int8_t*)outputPtr, (Index)rowsB,
      (const Index*)colIndexListPtr,
      (const Index*)colIndexListPtr + sizeColIndexList);
  fprintf(stderr, "Done Int8::SelectColumnsB\n");
  return 0;
}
