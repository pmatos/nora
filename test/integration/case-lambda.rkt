;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () () ((case-lambda ((x) x) ((x y) (+ x y))) 2))
