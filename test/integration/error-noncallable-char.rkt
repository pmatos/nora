;; RUN: not norac %s 2>&1 | FileCheck %s
;; Applying a non-procedure char immediate is an error, not a silent abort:
;; Value::operator bool() already treats any engaged immediate as "engaged"
;; (mirrors error-noncallable-boolean.rkt).
;; CHECK: error: application: expected a procedure in operator position
(linklet () () (#\a 1 2))
