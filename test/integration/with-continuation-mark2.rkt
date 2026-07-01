;; RUN: norac %s | FileCheck %s
;; A mark installed before a call is visible from inside the callee, and
;; continuation-mark-set->list collects it.
;; CHECK: (1)
(linklet () ()
  (define-values (f)
    (lambda (x) (continuation-mark-set->list (current-continuation-marks) 'k)))
  (with-continuation-mark 'k 1 (f 0)))
