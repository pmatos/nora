;; RUN: norac %s | FileCheck %s
;;      CHECK: -1
;; CHECK-NEXT: 5
;; CHECK-NEXT: 3
(linklet () () (values -1 (+ 2 3) 3))