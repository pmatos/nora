;; RUN: norac %s | FileCheck %s
;; Two nested with-continuation-mark forms for the same key: continuation-mark-
;; set-first returns the innermost (most recent) value, so the inner mark shadows
;; the outer one.
;; CHECK: 2
(linklet () ()
  (with-continuation-mark 'k 1
    (with-continuation-mark 'k 2
      (continuation-mark-set-first (current-continuation-marks) 'k))))
