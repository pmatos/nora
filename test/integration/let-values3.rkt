;; RUN: norac %s | FileCheck %s
;; CHECK: 8
(linklet () ()
  (let-values ([(x) 5])
    (let-values ([(f) (lambda (y) (+ x y))])
      (let-values ([(x) 7])
        (f 3)))))
