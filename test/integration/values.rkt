;; RUN: norac %s | FileCheck %s
;;      CHECK: 1
;; CHECK-NEXT: 2
;; CHECK-NEXT: 3
(linklet () () (values 1 2 3))