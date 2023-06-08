;; RUN: norac %s | FileCheck %s
;; CHECK: 1
(linklet () ()
  (define-values (x) (values 0))
  (let-values ((() (values)))
    (set! x 1))
  x)
