;; RUN: norac %s | FileCheck %s
;; CHECK: 3
(linklet () ()
  (let-values (((x y) (values 1 2)))
    (+ x y)))
