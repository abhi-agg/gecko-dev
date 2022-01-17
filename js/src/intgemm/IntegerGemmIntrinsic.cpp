/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "intgemm/IntegerGemmIntrinsic.h"
#include "mozilla/Logging.h"
#include <intgemm.h>

#include <utility>

#include "vm/JSContext.h"
#include "wasm/WasmInstance.h"

#include "vm/ArrayBufferObject-inl.h"

using namespace js::wasm;

namespace mozilla {
static mozilla::LazyLogModule gIntegerGemmLog("IntegerGemmLog");
#define LOG(level, ...) \
  MOZ_LOG(gIntegerGemmLog, mozilla::LogLevel::level, (__VA_ARGS__))
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

void ReportError(const unsigned errorNumber) {
  JSContext* cx = TlsContext.get();
  JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr, errorNumber);
}

size_t getWasmRawBufferLength(const uint8_t* memBase) {
#if INTGEMM_SHARED_MEMORY
  // TODO: Be more careful with using shared buffer
  const js::SharedArrayRawBuffer* rawBuf =
      js::SharedArrayRawBuffer::fromDataPtr(memBase);
  return rawBuf->volatileByteLength();
#else
  const js::WasmArrayRawBuffer* rawBuf =
      js::WasmArrayRawBuffer::fromDataPtr(memBase);
  return rawBuf->byteLength();
#endif
}

bool isMemoryBoundCheckPassed(uint32_t input, uint64_t inputSize,
                              const uint8_t* memBase) {
  size_t wasmBufferLimit = getWasmRawBufferLength(memBase);
  uint64_t inputUpperLimit = (uint64_t)input + inputSize;
  bool overflow = (inputUpperLimit < inputSize);
  return !overflow && (inputUpperLimit < wasmBufferLimit);
}

bool isAlignmentCheckPassed(const uint8_t* ptr) {
  return ((reinterpret_cast<uintptr_t>(ptr) % MAX_REGISTER_SIZE) == 0);
}

}  // namespace intgemm
}  // namespace js

int32_t js::intgemm::intrI8PrepareB(wasm::Instance* instance,
                                    uint32_t inputMatrixB, float scale,
                                    float zeroPoint, uint32_t rowsB,
                                    uint32_t colsB, uint32_t outputMatrixB,
                                    uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixB:%x" PRIu32
          "  scale:%f  zeroPoint:%f  "
          "rowsB:%" PRIu32 "  colsB:%" PRIu32
          "  outputMatrixB:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixB, scale, zeroPoint,
          rowsB, colsB, outputMatrixB);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixB, matrixSize, memBase) ||
      !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPtr = &memBase[inputMatrixB];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixBPtr) ||
      !isAlignmentCheckPassed(outputMatrixBPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

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
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBTransposed, float scale,
    float zeroPoint, uint32_t rowsB, uint32_t colsB, uint32_t outputMatrixB,
    uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixBTransposed:%x" PRIu32
          "  scale:%f  zeroPoint:%f  "
          "rowsB:%" PRIu32 "  colsB:%" PRIu32
          "  outputMatrixB:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixBTransposed, scale, zeroPoint,
          rowsB, colsB, outputMatrixB);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixBTransposed, matrixSize, memBase) ||
      !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBTransposedPtr = &memBase[inputMatrixBTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixBTransposedPtr) ||
      !isAlignmentCheckPassed(outputMatrixBPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm) for PrepareBTransposed
  fprintf(stderr,
          "%s: Bt:%p   Bp:%p   width:%" PRIu32 "   colsB:%" PRIu32
          "   Bt_align:%u   Bp_align:%u\n",
          __FUNCTION__, inputMatrixBTransposedPtr, outputMatrixBPtr, rowsB,
          colsB, computeAlignment((void*)inputMatrixBTransposedPtr),
          computeAlignment((void*)outputMatrixBPtr));
  ::intgemm::Int8::PrepareBTransposed((const float*)inputMatrixBTransposedPtr,
                                      (int8_t*)outputMatrixBPtr,
                                      (float)scale,  // Quant Mult
                                      rowsB, colsB);
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromQuantizedTransposed(
    wasm::Instance* instance, uint32_t inputMatrixBQuantizedTransposed,
    uint32_t rowsB, uint32_t colsB, uint32_t outputMatrixB, uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixBQuantizedTransposed:%x" PRIu32
          "  rowsB:%" PRIu32 "  colsB:%" PRIu32
          "  outputMatrixB:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixBQuantizedTransposed,
          rowsB, colsB, outputMatrixB);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixBQuantizedTransposed, matrixSize,
                                memBase) ||
      !isMemoryBoundCheckPassed(outputMatrixB, matrixSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBQuantizedTransposedPtr =
      &memBase[inputMatrixBQuantizedTransposed];
  uint8_t* outputMatrixBPtr = &memBase[outputMatrixB];

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixBQuantizedTransposedPtr) ||
      !isAlignmentCheckPassed(outputMatrixBPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
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
  return 0;
}

int32_t js::intgemm::intrI8PrepareA(wasm::Instance* instance,
                                    uint32_t inputMatrixA, float scale,
                                    float zeroPoint, uint32_t rowsA,
                                    uint32_t colsA, uint32_t outputMatrixA,
                                    uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixA:%x" PRIu32
          "  scale:%f  zeroPoint:%f  "
          "rowsA:%" PRIu32 "  colsA:%" PRIu32
          "  outputMatrixA:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixA, scale, zeroPoint,
          rowsA, colsA, outputMatrixA);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsA * (uint64_t)colsA;
  if ((matrixSize == 0) || (rowsA % ROWS_A_MULTIPLIER != 0) ||
      (colsA % COLUMNS_A_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixA, matrixSize, memBase) ||
      !isMemoryBoundCheckPassed(outputMatrixA, matrixSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPtr = &memBase[inputMatrixA];
  uint8_t* outputMatrixAPtr = &memBase[outputMatrixA];

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixAPtr) ||
      !isAlignmentCheckPassed(outputMatrixAPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  fprintf(stderr,
          "\n%s: A:%p   Ap:%p   "
          "rowsA:%" PRIu32 "   width:%" PRIu32 "   A_align:%u   Ap_align:%u\n",
          __FUNCTION__, inputMatrixAPtr, outputMatrixAPtr, rowsA, colsA,
          computeAlignment((void*)inputMatrixAPtr),
          computeAlignment((void*)outputMatrixAPtr));
  ::intgemm::Int8Shift::PrepareA((const float*)inputMatrixAPtr,
                                 (int8_t*)outputMatrixAPtr, scale, rowsA,
                                 colsA);
  return 0;
}

int32_t js::intgemm::intrI8PrepareBias(
    wasm::Instance* instance, uint32_t inputMatrixBPrepared, float scaleA,
    float zeroPointA, float scaleB, float zeroPointB, uint32_t rowsB,
    uint32_t colsB, uint32_t inputBias, uint32_t output, uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixBPrepared:%x" PRIu32
          "  scaleA:%f  zeroPointA:%f  "
          "  scaleB:%f  zeroPointB:%f  "
          "rowsB:%" PRIu32 "  colsB:%" PRIu32
          "  inputBias:%x" PRIu32 "  output:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixBPrepared, scaleA, zeroPointA,
          scaleB, zeroPointB, rowsB, colsB, inputBias, output);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  if ((matrixSize == 0) || (rowsB % ROWS_B_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize, memBase) ||
      !isMemoryBoundCheckPassed(inputBias, colsB, memBase) ||
      !isMemoryBoundCheckPassed(output, colsB, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPtr = &memBase[inputBias];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor =
      (-1) * ((127.0f / scaleA) * (127.0f / scaleB)) / (127.0f);

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  fprintf(stderr,
          "\n%s: Bp:%p   bias:%p   bias_p:%p   "
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
  return 0;
}

int32_t js::intgemm::intrI8MultiplyAndAddBias(
    wasm::Instance* instance, uint32_t inputMatrixAPrepared, float scaleA,
    float zeroPointA, uint32_t inputMatrixBPrepared, float scaleB,
    float zeroPointB, uint32_t inputBiasPrepared, float unquantMultiplier,
    uint32_t rowsA, uint32_t width, uint32_t colsB, uint32_t output,
    uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixAPrepared:%x" PRIu32
          "  scaleA:%f  zeroPointA:%f  "
          "inputMatrixBPrepared:%x" PRIu32
          "  scaleB:%f  zeroPointB:%f  "
          "inputBiasPrepared:%x" PRIu32
          "  unquantMultiplier:%f  "
          "rowsA:%" PRIu32 "  width:%" PRIu32 "  colsB:%" PRIu32
          "  output:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixAPrepared, scaleA, zeroPointA,
          inputMatrixBPrepared, scaleB, zeroPointB, inputBiasPrepared,
          unquantMultiplier, rowsA, width, colsB, output);

  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixASize = (uint64_t)rowsA * (uint64_t)width;
  uint64_t matrixBSize = (uint64_t)width * (uint64_t)colsB;
  uint64_t inputBiasSize = (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsA * (uint64_t)colsB;
  if ((matrixASize == 0) || (matrixBSize == 0) ||
      (rowsA % ROWS_A_MULTIPLIER != 0) || (width % COLUMNS_A_MULTIPLIER != 0) ||
      (colsB % COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixAPrepared, matrixASize, memBase) ||
      !isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixBSize, memBase) ||
      !isMemoryBoundCheckPassed(inputBiasPrepared, inputBiasSize, memBase) ||
      !isMemoryBoundCheckPassed(output, outputSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixAPreparedPtr = &memBase[inputMatrixAPrepared];
  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* inputBiasPreparedPtr = &memBase[inputBiasPrepared];
  uint8_t* outputPtr = &memBase[output];
  float unquantFactor = unquantMultiplier / (scaleA * scaleB);

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixAPreparedPtr) ||
      !isAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
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
  return 0;
}

int32_t js::intgemm::intrI8SelectColumnsOfB(wasm::Instance* instance,
                                            uint32_t inputMatrixBPrepared,
                                            uint32_t rowsB, uint32_t colsB,
                                            uint32_t colIndexList,
                                            uint32_t sizeColIndexList,
                                            uint32_t output, uint8_t* memBase) {
  fprintf(stderr,
          "\n%s: inputMatrixBPrepared:%x" PRIu32
          "  rowsB:%" PRIu32 "  colsB:%" PRIu32
          "  colIndexList:%x" PRIu32 "  sizeColIndexList:%" PRIu32
          "  output:%x" PRIu32 "\n",
          __FUNCTION__, inputMatrixBPrepared, rowsB, colsB,
          colIndexList, sizeColIndexList, output);
  MOZ_ASSERT(SASigIntrI8PrepareB.failureMode == FailureMode::FailOnNegI32);

  // Size checks for matricies
  uint64_t matrixSize = (uint64_t)rowsB * (uint64_t)colsB;
  uint64_t outputSize = (uint64_t)rowsB * (uint64_t)sizeColIndexList;
  if ((matrixSize == 0) || (outputSize == 0) ||
      (rowsB % ROWS_B_MULTIPLIER != 0) || (colsB % COLUMNS_B_MULTIPLIER != 0) ||
      (sizeColIndexList % SELECTED_COLUMNS_B_MULTIPLIER != 0)) {
    ReportError(JSMSG_WASM_UNREACHABLE);
    return -1;
  }

  // Memory Bound checks for all matricies
  if (!isMemoryBoundCheckPassed(inputMatrixBPrepared, matrixSize, memBase) ||
      !isMemoryBoundCheckPassed(colIndexList, sizeColIndexList, memBase) ||
      !isMemoryBoundCheckPassed(output, outputSize, memBase)) {
    ReportError(JSMSG_WASM_OUT_OF_BOUNDS);
    return -1;
  }

  uint8_t* inputMatrixBPreparedPtr = &memBase[inputMatrixBPrepared];
  uint8_t* colIndexListPtr = &memBase[colIndexList];
  uint8_t* outputPtr = &memBase[output];

  // Pointer Alignment checks for matricies
  if (!isAlignmentCheckPassed(inputMatrixBPreparedPtr)) {
    ReportError(JSMSG_WASM_UNALIGNED_ACCESS);
    return -1;
  }

  // Actual call to the 3rd party library (intgemm)
  ::intgemm::Int8::SelectColumnsB(
      (const int8_t*)inputMatrixBPreparedPtr, (int8_t*)outputPtr, rowsB,
      (const uint32_t*)colIndexListPtr,
      (const uint32_t*)colIndexListPtr + sizeColIndexList);
  return 0;
}
