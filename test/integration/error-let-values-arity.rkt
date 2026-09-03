;; RUN: not norac %s 2>&1 | FileCheck %s
;; Binding N identifiers to a values result of a different arity is an error.
;; Pins the let-values diagnostic wording ahead of the continuation refactor.
;; CHECK: error: let-values binding expected 2 values, got 3
(linklet () () (let-values ([(x y) (values 1 2 3)]) x))
