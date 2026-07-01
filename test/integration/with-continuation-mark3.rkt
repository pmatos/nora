;; RUN: norac %s | FileCheck %s
;; Two with-continuation-mark forms for the same key in the same continuation
;; frame (nested in tail position): the inner value overwrites the outer one.
;; CHECK: 2
(linklet () ()
  (with-continuation-mark 'k 1
    (with-continuation-mark 'k 2
      (continuation-mark-set-first (current-continuation-marks) 'k))))
