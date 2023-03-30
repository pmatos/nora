;; RUN: norac %s | FileCheck %s
;; CHECK: 43
(linklet () () 
    (define-values (fn x) (values (lambda (x) (+ x 1)) 2)) 
    (set! x 42)
    (fn x))
