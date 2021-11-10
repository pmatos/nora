;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () () (+ 2 0))
