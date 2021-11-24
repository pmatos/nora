#lang racket/base

;; RUN: %nora %s | filecheck %s
;;      CHECK: (module const racket/base
;; CHECK-NEXT:  (#%module-begin
;; CHECK-NEXT:   (module configure-runtime '#%kernel
;; CHECK-NEXT:     (#%module-begin (#%require racket/runtime-config) (#%app configure '#f)))
;; CHECK-NEXT:   (#%app call-with-values (lambda () '0) print-values))

0
