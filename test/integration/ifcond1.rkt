;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () () 
  (define-values (x) (values 1))
  (if x (+ x x) x))
