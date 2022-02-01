// |jit-test| skip-if:true;

// This file contains all the tests for int8_prepare_b_from_transposed intrinsic. All these
// tests are run by main.js file in this directory.
let {int8_prepare_b_from_transposed} = instance.exports;

const VALID = {input: 0, scale: 1.0, zeroPoint: 0.0, rows: ROWS_B_MULTIPLIER, cols: COLUMNS_B_MULTIPLIER, output: 1024};

function testInvalidSize() {
  var invalid;

  // row: 0
  invalid = 0;
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, invalid, VALID.cols, VALID.output), WebAssembly.RuntimeError, /unreachable/);

  // row: Not an integral multiple of ROWS_B_MULTIPLIER
  invalid = ROWS_B_MULTIPLIER + 1;
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, invalid, VALID.cols, VALID.output), WebAssembly.RuntimeError, /unreachable/);

  // col: 0
  invalid = 0;
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, VALID.rows, invalid, VALID.output), WebAssembly.RuntimeError, /unreachable/);

  // col: Not an integral multiple of COLUMNS_B_MULTIPLIER
  invalid = COLUMNS_B_MULTIPLIER + 1;
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, VALID.rows, invalid, VALID.output), WebAssembly.RuntimeError, /unreachable/);
}

function testInvalidAlignment() {
  var invalid = ARRAY_ALIGNMENT + 1;

  // input: Not an integral multiple of ARRAY_ALIGNMENT
  assertErrorMessage(() => int8_prepare_b_from_transposed(invalid, VALID.scale, VALID.zeroPoint, VALID.rows, VALID.cols, VALID.output), WebAssembly.RuntimeError, /index out of bounds/);

  // output: Not an integral multiple of ARRAY_ALIGNMENT
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, VALID.rows, VALID.cols, invalid), WebAssembly.RuntimeError, /index out of bounds/);
}

function testOutOfBounds() {
  var outOfBound = PageSizeInBytes - ARRAY_ALIGNMENT;

  // input: Out of Bounds
  assertErrorMessage(() => int8_prepare_b_from_transposed(outOfBound, VALID.scale, VALID.zeroPoint, VALID.rows, VALID.cols, VALID.output), WebAssembly.RuntimeError, /index out of bounds/);

  // output: Out of Bounds
  assertErrorMessage(() => int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, VALID.rows, VALID.cols, outOfBound), WebAssembly.RuntimeError, /index out of bounds/);
}

function testSuccessfulCall() {
  let buffer = new Int8Array(memory.buffer);
  let size = VALID.rows * VALID.cols;
  for (let i = 0; i < size; i++) {
    buffer[i + VALID.input] = i + VALID.input;
    buffer[i + VALID.output] = i + VALID.output;
  }
  int8_prepare_b_from_transposed(VALID.input, VALID.scale, VALID.zeroPoint, VALID.rows, VALID.cols, VALID.output);
}

testInvalidSize();
testInvalidAlignment();
testOutOfBounds();
testSuccessfulCall();
