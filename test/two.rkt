#lang racket/base

;; RUN: norac %s | filecheck %s
;;      CHECK: (module two racket/base
;; CHECK-NEXT:  (#%module-begin
;; CHECK-NEXT:   (module configure-runtime '#%kernel
;; CHECK-NEXT:     (#%module-begin (#%require racket/runtime-config) (#%app configure '#f)))
;; CHECK-NEXT:   (#%app call-with-values (lambda () (#%app + '1 '1)) print-values)))
(+ 1 1)
