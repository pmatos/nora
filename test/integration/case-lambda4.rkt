;; RUN: norac %s | FileCheck %s
;; A case-lambda captures its free variables across every clause.
;; CHECK: 15
(linklet () ()
  (define-values (f) (values 0))
  (let-values (((n) (values 10)))
    (set! f (case-lambda ((x) (+ x n)) ((x y) (+ x y n)))))
  (f 5))
