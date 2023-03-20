;; RUN: norac %s | FileCheck %s
;; CHECK: 3
(linklet () () (begin 1 2 3))

