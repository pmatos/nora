;; RUN: norac %s | FileCheck %s
;; Marks for the same key set in different continuation frames (a caller and a
;; callee) accumulate; continuation-mark-set->list returns them innermost first.
;; CHECK: (2 1)
(linklet () ()
  (define-values (f)
    (lambda (x)
      (with-continuation-mark 'k 2
        (continuation-mark-set->list (current-continuation-marks) 'k))))
  (with-continuation-mark 'k 1 (f 0)))
