;; RUN: norac %s | FileCheck %s
;; CHECK: (2 3)
(linklet () () 
  ((lambda (x . y) y) 1 2 3))
