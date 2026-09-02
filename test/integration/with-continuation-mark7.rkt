;; RUN: norac %s | FileCheck %s
;; Marks for the same key set in genuinely different continuation frames (a
;; caller and a callee) accumulate; continuation-mark-set->list returns them
;; innermost first. Unlike with-continuation-mark4.rkt, (f 0) here is bound by
;; let-values rather than called in tail position, so it does NOT reuse the
;; outer with-continuation-mark's frame.
;; CHECK: (2 1)
(linklet () ()
  (define-values (f)
    (lambda (x)
      (with-continuation-mark 'k 2
        (continuation-mark-set->list (current-continuation-marks) 'k))))
  (with-continuation-mark 'k 1
    (let-values ([(r) (f 0)]) r)))
