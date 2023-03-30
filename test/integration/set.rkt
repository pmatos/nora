;; RUN: norac %s | FileCheck %s
;; CHECK: 42
(linklet () () 
    (define-values (x) (values 2))
    (set! x 42)
    x)
