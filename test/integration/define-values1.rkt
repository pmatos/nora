;; RUN: norac %s | FileCheck %s
;;      CHECK: 1
;; CHECK-NEXT: 4
;; CHECK-NEXT: 1
(linklet () () 
  (define-values (x y) (values 1 2))
  (values x (+ y y) x))
