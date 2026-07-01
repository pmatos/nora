;; RUN: norac %s | FileCheck %s
;; An identifier formal accepts any number of arguments as a list, so it acts
;; as the fallthrough clause after the fixed-arity clauses.
;; CHECK: (1 2)
(linklet () () ((case-lambda (() 0) (args args)) 1 2))
