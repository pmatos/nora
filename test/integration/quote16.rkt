;; RUN: norac %s | FileCheck %s
;; CHECK: '(#t 1 #f 2)
(linklet () () '(#t 1 #f 2))
