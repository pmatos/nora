;; RUN: norac %s | FileCheck %s
;; CHECK: 105
;; A closure created in an earlier clause can refer to an identifier bound by a
;; later clause; the reference resolves when the closure is applied.
(linklet () ()
  (letrec-values ([(f) (lambda (n) (+ n c))]
                  [(c) 100])
    (f 5)))
