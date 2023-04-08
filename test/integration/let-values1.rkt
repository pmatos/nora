;; RUN: norac %s | FileCheck %s
;; CHECK: 6
(linklet () ()
  (let-values (((x y z) (values 1 2 3)))
    (+ x (+ y z))))
