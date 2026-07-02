;; RUN: not norac %s 2>&1 | FileCheck %s
;; Applying a non-procedure value is an error.
;; CHECK: error: application: expected a procedure in operator position
(linklet () () (1 2))
