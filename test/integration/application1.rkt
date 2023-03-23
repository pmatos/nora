;; RUN: norac %s | FileCheck %s
;; CHECK: (2 3)
(linklet () () 
  ((lambda x x) 2 3))
