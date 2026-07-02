;; RUN: not norac %s 2>&1 | FileCheck %s
;; Applying a lambda with the wrong number of arguments is an arity error.
;; CHECK: error: arity mismatch: expected 1 argument(s), got 2
(linklet () () ((lambda (x) x) 1 2))
