;; RUN: norac %s | FileCheck %s
;; The clause selected depends on the number of arguments.
;; CHECK: 7
(linklet () () ((case-lambda ((x) x) ((x y) (+ x y))) 3 4))
