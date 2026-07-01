;; RUN: norac %s | FileCheck %s
;; CHECK: 20
;; Unlike let-values, a letrec-values binding is in scope for the binding
;; expressions that follow it, so y's expression can refer to x.
(linklet () ()
  (letrec-values ([(x) 5]
                  [(y) (+ x 10)])
    (+ x y)))
