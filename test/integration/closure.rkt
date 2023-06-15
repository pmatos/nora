;; RUN: norac %s | FileCheck %s
;; CHECK: 12
(linklet () ()
  (define-values (fn) (values 0))
  (let-values (((x) (values 2)))
     (set! fn (lambda (y) (+ x y))))
  (let-values (((x) (values 3)))
     (fn 10)))
