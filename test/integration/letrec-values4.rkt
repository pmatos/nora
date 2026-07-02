;; RUN: norac %s | FileCheck %s
;; CHECK: 15
;; A closure encloses a letrec-values whose binding expression captures a free
;; variable (base) from the surrounding lambda. This exercises free-variable
;; analysis over the letrec binding expressions, so the closure built for the
;; inner letrec captures base correctly.
(linklet () ()
  (let-values ([(make) (lambda (base)
                         (letrec-values ([(f) (lambda (n) (+ n base))])
                           (f 10)))])
    (make 5)))
