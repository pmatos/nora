;; RUN: norac %s | FileCheck %s
;; CHECK: 30
(linklet () ()
  (let-values (((x) (values 10)))
    (let-values (((y) (values 20)))
      (+ x y))))
