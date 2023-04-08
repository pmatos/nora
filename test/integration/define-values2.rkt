;; RUN: norac %s | FileCheck %s
;;      CHECK: 3
(linklet () () 
  (define-values (x y) (values 1 2))
  (+ x y))