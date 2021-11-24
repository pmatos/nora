#lang racket/base

;; RUN: norac %s | filecheck %s
;;      CHECK: (module printf-two racket/base
;; CHECK-NEXT:   (#%module-begin
;; CHECK-NEXT:    (module configure-runtime '#%kernel
;; CHECK-NEXT:     (#%module-begin (#%require racket/runtime-config) (#%app configure '#f)))
;; CHECK-NEXT:   (#%app
;; CHECK-NEXT:    call-with-values
;; CHECK-NEXT:    (lambda () (#%app printf '"~a" (#%app + '1 '1)))
;; CHECK-NEXT:    print-values)))

(printf "~a" (+ 1 1))
