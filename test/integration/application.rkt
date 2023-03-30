;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () () ((lambda (x) x) 2))
