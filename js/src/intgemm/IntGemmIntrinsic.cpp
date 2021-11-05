#include "intgemm/IntGemmIntrinsic.h"
#include <intgemm.h>

#include <utility>

#include "vm/JSContext.h"
#include "wasm/WasmInstance.h"

#include "vm/ArrayBufferObject-inl.h"

using namespace js::wasm;

#define INTGEMM_INTR_SHARED 1

int32_t js::intgemm::intrSample1(Instance* instance, uint32_t arr,
                                          uint32_t len, uint8_t* memBase) {
  MOZ_ASSERT(SASigIntrSample1.failureMode == FailureMode::FailOnNegI32);

  const int8_t* A = (int8_t*)10;
  int8_t* output_addr = (int8_t*)calloc(10, sizeof(int8_t));;
  ::intgemm::Int8::PrepareBQuantizedTransposed(A, output_addr, 5, 6);

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

/*

void Int8ShiftedMultiplyExport(const int8_t* A, const int8_t* B, int A_rows,
                               int width, int B_cols, float unquant_mult,
                               const float* bias_addr, float* output_addr) {
  intgemm::Int8Shift::Multiply(A, B, A_rows, width, B_cols,
                               intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
                                   unquant_mult, bias_addr, output_addr));
}

void Int8PrepareAExport(const float* A, int8_t* output_addr, float quant_mult,
                        int rows, int cols) {
  intgemm::Int8::PrepareA(A, output_addr, quant_mult, rows, cols);
}

void Int8ShiftedPrepareAExport(const float* A, int8_t* output_addr,
                               float quant_mult, int rows, int cols) {
  intgemm::Int8Shift::PrepareA(A, output_addr, quant_mult, rows, cols);
}

void Int8PrepareBExport(const float* A, int8_t* output_addr, float quant_mult,
                        int rows, int cols) {
  intgemm::Int8::PrepareB(A, output_addr, quant_mult, rows, cols);
}

void Int8ShiftedPrepareBiasExport(const int8_t* A, int rows, int cols,
                                  float unquant_mult, const float* bias_addr,
                                  float* output_addr) {
  intgemm::Int8Shift::PrepareBias(
      A, rows, cols,
      intgemm::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias_addr,
                                                       output_addr));
}

void Int8MultiplyExport(const int8_t* A, const int8_t* B, int A_rows, int width,
                        int B_cols, float unquant_mult, float* output_addr) {
  intgemm::Int8::Multiply(
      A, B, A_rows, width, B_cols,
      intgemm::callbacks::UnquantizeAndWrite(unquant_mult, output_addr));
}

void Int8PrepareBQuantizedTransposedExport(const int8_t* A, int8_t* output_addr,
                                           int rows, int cols) {
  intgemm::Int8::PrepareBQuantizedTransposed(A, output_addr, rows, cols);
}
*/


int32_t js::intgemm::intrI8PrepareB(wasm::Instance* instance,
             uint32_t inputMatrixB,
             float scale,
             float zeroPoint,
             uint32_t rowsB,
             uint32_t colsB,
             uint32_t outputMatrixB,
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

  // Actual call to the 3rd party library (intgemm) for Prepare
  ::intgemm::Int8::PrepareB((const float*)inputMatrixB,
                          (int8_t*)outputMatrixB,
                          (float)scale, /*Quant Mult*/
                          (Index)rowsB,
                          (Index)colsB);
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromTransposed(wasm::Instance* instance,
             uint32_t inputMatrixBTransposed,
             float scale,
             float zeroPoint,
             Index rowsB,
             Index colsB,
             uint32_t outputMatrixB,
             uint8_t* memBase) {
  // ToDo: Write implementation
  return 0;
}

int32_t js::intgemm::intrI8PrepareBFromQuantizedTransposed(wasm::Instance* instance,
             uint32_t inputMatrixBQuantizedTransposed,
             Index rowsB,
             Index colsB,
             uint32_t outputMatrixB,
             uint8_t* memBase) {
  // ToDo: Write implementation
  return 0;
}

int32_t js::intgemm::intrI8PrepareA(wasm::Instance* instance,
             uint32_t inputMatrixA,
             float scale,
             float zeroPoint,
             Index rowsA,
             Index colsA,
             uint32_t outputMatrixA,
             uint8_t* memBase) {
  // ToDo: Write implementation
  return 0;
}