;; RUN: not norac %s 2>&1 | FileCheck %s
;; letrec-values shares let-values' multiple-values binder, so it reports the
;; same wording on an arity mismatch. Pinned ahead of the continuation refactor.
;; CHECK: error: let-values binding expected 2 values, got 3
(linklet () () (letrec-values ([(x y) (values 1 2 3)]) x))
