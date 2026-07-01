;; RUN: norac %s | FileCheck %s
;; CHECK: 7
;; Mutually referring closures: f calls g, which is bound in a later clause.
(linklet () ()
  (letrec-values ([(f) (lambda (n) (g n))]
                  [(g) (lambda (n) (+ n 2))])
    (f 5)))
