;; RUN: norac %s | FileCheck %s
;; A with-continuation-mark in non-tail position (here, the first expression of
;; a begin) must not leak its mark into later expressions: once it returns, the
;; mark is out of its dynamic extent and continuation-mark-set-first sees #f.
;; CHECK: #f
(linklet () ()
  (begin (with-continuation-mark 'k 1 0)
         (continuation-mark-set-first (current-continuation-marks) 'k)))
