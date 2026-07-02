// RUN: nir-opt %s | nir-opt | FileCheck %s
//
// Round-trips the minimal NIR skeleton (parse -> verify -> print, twice) to
// prove the dialect is registered and its assembly format is stable. This is
// the R5 acceptance test. It is NOT wired into ctest yet (nir-opt only exists
// under the `mlir` preset); run it manually once MLIR is installed:
//
//   cmake --preset mlir && cmake --build --preset mlir --target nir-opt
//   build/mlir/bin/nir-opt test/mlir/nir-roundtrip.mlir | build/mlir/bin/nir-opt

// CHECK-LABEL: func.func @const_return
func.func @const_return() -> i64 {
  // CHECK: %[[C:.*]] = nir.constant 42 : i64
  %0 = nir.constant 42 : i64
  // CHECK: return %[[C]] : i64
  return %0 : i64
}
