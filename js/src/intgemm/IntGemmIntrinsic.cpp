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
#define LOG(level, ...) \
  MOZ_LOG(gIntGemmLog, mozilla::LogLevel::level, (__VA_ARGS__))
}  // namespace mozilla

namespace js {
namespace intgemm {

unsigned computeAlignment(void* address) {
  auto ptr = reinterpret_cast<std::uintptr_t>(address);
  if ((ptr % 512) == 0) {
    return 512;
  } else if ((ptr % 256) == 0) {
    return 256;
  } else if ((ptr % 128) == 0) {
    return 128;
  } else if ((ptr % 64) == 0) {
    return 64;
  } else if ((ptr % 32) == 0) {
    return 32;
  } else if ((ptr % 16) == 0) {
    return 16;
  } else if ((ptr % 8) == 0) {
    return 8;
  } else if ((ptr % 4) == 0) {
    return 4;
  } else if ((ptr % 2) == 0) {
    return 2;
  } else {
    return 1;
  }
}

#define INTGEMM_INTR_SHARED 0

size_t getWasmRawBufferLength(uint8_t* memBase) {
#if INTGEMM_INTR_SHARED
  const js::SharedArrayRawBuffer* rawBuf =
      js::SharedArrayRawBuffer::fromDataPtr(memBase);
  return rawBuf->volatileByteLength();
  // TODO shall be more carefull with using shared buffer
#else
  const js::WasmArrayRawBuffer* rawBuf =
      js::WasmArrayRawBuffer::fromDataPtr(memBase);
  return rawBuf->byteLength();
#endif
}

bool isMemoryBoundCheckPassed(uint32_t input, uint64_t inputSize,
                              size_t wasmBufferLimit) {
  // ToDo: Deal with arithmetic overflow
  uint64_t inputUpperLimit = (uint64_t)input + inputSize;
  return (inputUpperLimit > wasmBufferLimit) ? false : true;
}
}  // namespace intgemm
}  // namespace js

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
                                    float zeroPoint, Size rowsB, Size colsB,
                                    uint32_t outputMatrixB, uint8_t* memBase) {
  fprintf(stderr,
          "intrI8PrepareB called with inputMatrixB:%d outputMatrixB:%d\n",
          inputMatrixB, outputMatrixB);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixB, matrixSize, wasmBufferLen) ||
      !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPtr = &memBase[inputMatrixB];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];
  // Actual call to the 3rd party library (intgemm) for PrepareB
  fprintf(stderr,
          "%s: B:%p   Bp:%p   width:%" PRIu32 "   colsB:%" PRIu32
          "   B_align:%u   Bp_align:%u\n",
          __FUNCTION__, inputMatrixBPtr, outputMatrixBPtr, rowsB, colsB,
          computeAlignment((void*)inputMatrixBPtr),
          computeAlignment((void*)outputMatrixBPtr));
  ::intgemm::Int8::PrepareB((const float*)inputMatrixBPtr,
                            (int8_t*)outputMatrixBPtr,
                            (float)scale,  // Quant Mult
                            rowsB, colsB);
  fprintf(stderr, "Done Int8::PrepareB\n");
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBTransposed, float scale,
    float zeroPoint, Size rowsB, Size colsB, uint32_t outputMatrixB,
    uint8_t* memBase) {
  JSContext* cx = TlsContext.get();
  JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                            JSMSG_WASM_OUT_OF_BOUNDS);
  return -1;
}

int32_t js::intgemm::intrI8PrepareBFromQuantizedTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBQuantizedTransposed,
    Size rowsB, Size colsB, uint32_t outputMatrixB, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixBQuantizedTransposed, matrixSize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBQuantizedTransposedPtr =
      &memBase[inputMatrixBQuantizedTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];
  fprintf(
      stderr,
      "%s: Bqt:%p   Bp:%p   "
      "width:%" PRIu32 "   colsB:%" PRIu32 "   Bqt_align:%u   Bp_align:%u\n",
      __FUNCTION__, inputMatrixBQuantizedTransposedPtr, outputMatrixBPtr, rowsB,
      colsB, computeAlignment((void*)inputMatrixBQuantizedTransposedPtr),
      computeAlignment((void*)outputMatrixBPtr));
  ::intgemm::Int8::PrepareBQuantizedTransposed(
      (const int8_t*)inputMatrixBQuantizedTransposedPtr,
      (int8_t*)outputMatrixBPtr, rowsB, colsB);
  fprintf(stderr, "Done Int8::PrepareBQuantizedTransposed\n");
  return 0;
}

int32_t js::intgemm::intrI8PrepareA(wasm::Instance* instance,
                                    uint32_t inputMatrixA, float scale,
                                    float zeroPoint, Size rowsA, Size colsA,
                                    uint32_t outputMatrixA, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsA * (uint64_t)colsA;
  if (!isMemoryBoundCheckPassed(inputMatrixA, matrixSize, wasmBufferLen) ||
      !isMemoryBoundCheckPassed(outputMatrixA, matrixSize, wasmBufferLen)) {
    JSContext* cx = TlsContext.get();
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                              JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPtr = &memBase[inputMatrixA];
  uint8_t* outputMatrixAPtr = &memBase[outputMatrixA];
  fprintf(stderr,
          "%s: A:%p   Ap:%p   "
          "rowsA:%" PRIu32 "   width:%" PRIu32 "   A_align:%u   Ap_align:%u\n",
          __FUNCTION__, inputMatrixAPtr, outputMatrixAPtr, rowsA, colsA,
          computeAlignment((void*)inputMatrixAPtr),
          computeAlignment((void*)outputMatrixAPtr));
  ::intgemm::Int8Shift::PrepareA((const float*)inputMatrixAPtr,
                                 (int8_t*)outputMatrixAPtr, scale, rowsA,
                                 colsA);
  fprintf(stderr, "Done Int8Shift::PrepareA\n");
  return 0;
}

int32_t js::intgemm::intrI8PrepareBias(
    wasm::Instance* instance, uint32_t inputMatrixBPrepared, float scaleA,
    float zeroPointA, float scaleB, float zeroPointB, Size rowsB, Size colsB,
    uint32_t inputBias, uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(inputBias, colsB, wasmBufferLen) ||
      !isMemoryBoundCheckPassed(output, colsB, wasmBufferLen)) {
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
  fprintf(stderr,
          "%s: Bp:%p   bias:%p   bias_p:%p   "
          "unquantFactor:%f   width:%" PRIu32 "   colsB:%" PRIu32
          "   Bp_align:%u   bias_align:%u   bias_p_align:%u\n",
          __FUNCTION__, inputMatrixBPreparedPtr, inputBiasPtr, outputPtr,
          unquantFactor, rowsB, colsB,
          computeAlignment((void*)inputMatrixBPreparedPtr),
          computeAlignment((void*)inputBiasPtr),
          computeAlignment((void*)outputPtr));
  ::intgemm::Int8Shift::PrepareBias(
      (const int8_t*)inputMatrixBPreparedPtr, rowsB, colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPtr, (float*)outputPtr));
  fprintf(stderr, "Done Int8Shift::PrepareBias\n");
  return 0;
}

int32_t js::intgemm::intrI8MultiplyAndAddBias(
    wasm::Instance* instance, uint32_t inputMatrixAPrepared, float scaleA,
    float zeroPointA, uint32_t inputMatrixBPrepared, float scaleB,
    float zeroPointB, uint32_t inputBiasPrepared, float unquantMultiplier,
    Size rowsA, Size width, Size colsB, uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixASize = (uint64_t)rowsA * (uint64_t)width;
  uint64_t matrixBSize = (uint64_t)width * (uint64_t)colsB;
  uint64_t inputBiasSize = (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsA * (uint64_t)colsB;
  if (!isMemoryBoundCheckPassed(inputMatrixAPrepared, matrixASize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixBSize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(inputBiasPrepared, inputBiasSize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(output, outputSize, wasmBufferLen)) {
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
  fprintf(stderr,
          "%s: Ap:%p   Bp:%p   bias_p:%p   output:%p   "
          "unquantFactor:%f   rowsA:%" PRIu32 "   width:%" PRIu32
          "   colsB:%" PRIu32
          "   Ap_align:%u   Bp_align:%u   bias_p_align:%u   output_align:%u\n",
          __FUNCTION__, inputMatrixAPreparedPtr, inputMatrixBPreparedPtr,
          inputBiasPreparedPtr, outputPtr, unquantFactor, rowsA, width, colsB,
          computeAlignment((void*)inputMatrixAPreparedPtr),
          computeAlignment((void*)inputMatrixBPreparedPtr),
          computeAlignment((void*)inputBiasPreparedPtr),
          computeAlignment((void*)outputPtr));
  ::intgemm::Int8Shift::Multiply(
      (const int8_t*)inputMatrixAPreparedPtr,
      (const int8_t*)inputMatrixBPreparedPtr, rowsA, width, colsB,
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
          unquantFactor, (const float*)inputBiasPreparedPtr,
          (float*)outputPtr));
  fprintf(stderr, "Done Int8Shift::Multiply\n");
  return 0;
}

int32_t js::intgemm::intrI8SelectColumnsOfB(wasm::Instance* instance,
                                            uint32_t inputMatrixBPrepared,
                                            Size rowsB, Size colsB,
                                            Size colIndexList,
                                            Size sizeColIndexList,
                                            uint32_t output, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Bounds check for all matricies and output
  // ToDo: Check matrix size requirements
  size_t wasmBufferLen = getWasmRawBufferLength(memBase);
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsB * (uint64_t)sizeColIndexList;
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(colIndexList, sizeColIndexList,
                                wasmBufferLen) ||
      !isMemoryBoundCheckPassed(output, outputSize, wasmBufferLen)) {
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
      (const int8_t*)inputMatrixBPreparedPtr, (int8_t*)outputPtr, rowsB,
      (const Size*)colIndexListPtr,
      (const Size*)colIndexListPtr + sizeColIndexList);
  fprintf(stderr, "Done Int8::SelectColumnsB\n");
  return 0;
}
