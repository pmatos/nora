;; RUN: norac %s | FileCheck %s
;; CHECK: 1
(linklet () () 
  (if #t 1 2))
