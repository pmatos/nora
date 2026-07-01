;; RUN: norac %s | FileCheck %s
;; A clause with a rest formal collects the extra arguments into a list.
;; CHECK: (2 3)
(linklet () () ((case-lambda ((x) x) ((x . y) y)) 1 2 3))
