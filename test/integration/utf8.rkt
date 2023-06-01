;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () () 
    (define-values (β) (values 1))
    (+ β β))
