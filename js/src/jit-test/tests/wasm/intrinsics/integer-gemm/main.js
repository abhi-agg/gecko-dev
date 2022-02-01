// |jit-test| skip-if:true;

// This is the main file that tests all integer gemm intrinsics

// True if running on x86/x86-64 hardware
function nativeX86Shared() {
  var conf = getBuildConfiguration();
  if (conf.x64 || conf.x86)
      return true;
  return false;
}

if (!nativeX86Shared())
    quit(0);

// All integer gemm intrinsic test files should be added here. Each
// of these files test one intrinsic
let TEST_SCRIPTS = ["I8PrepareB.js", "I8PrepareBFromTransposed.js"];

// The test setup that is common to all integer gemm tests
let COMMON_TEST_SETUP = `
const libdir=${JSON.stringify(libdir)}; load(libdir + "wasm.js");
let memory = new WebAssembly.Memory({initial: 1, maximum: 1});
let module = WebAssembly.mozIntGemm();
if (!module) {
  throw new Error();
}
let instance = new WebAssembly.Instance(module, {"": {"memory": memory}});

const ARRAY_ALIGNMENT = 64;
const ROWS_A_MULTIPLIER = 1;
const COLUMNS_A_MULTIPLIER = 64;
const ROWS_B_MULTIPLIER = COLUMNS_A_MULTIPLIER;
const COLUMNS_B_MULTIPLIER = 8;
const SELECTED_COLUMNS_B_MULTIPLIER = 8;
`

// Run all the tests
TEST_SCRIPTS.forEach((test_script) => {
  let COMPLETE_TEST = COMMON_TEST_SETUP + read(scriptdir + test_script);
  const testEnvironment = newGlobal({newCompartment: true, systemPrincipal: true});
  testEnvironment.evaluate(COMPLETE_TEST);
});
