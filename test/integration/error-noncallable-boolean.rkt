;; RUN: not norac %s 2>&1 | FileCheck %s
;; Applying a non-procedure boolean immediate is an error, not a silent abort:
;; Value::operator bool() must treat an engaged immediate as "engaged" (not
;; Racket-falsy), or applyProcedure's `if (!Op)` misreports this as an
;; aborted/missing value instead of the diagnostic below.
;; CHECK: error: application: expected a procedure in operator position
(linklet () () (#f 1 2))
