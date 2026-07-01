;; RUN: norac %s | FileCheck %s
;; CHECK: 15
;; Companion to letrec-values4: a closure enclosing a plain let-values whose
;; binding expression captures a free variable (base). Confirms free-variable
;; analysis walks let-values binding expressions in the enclosing scope.
(linklet () ()
  (let-values ([(make) (lambda (base)
                         (let-values ([(f) (lambda (n) (+ n base))])
                           (f 10)))])
    (make 5)))
