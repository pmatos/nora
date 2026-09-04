;; RUN: not norac %s 2>&1 | FileCheck %s
;; define-values keeps its own distinct diagnostic wording (not let-values').
;; Pins that divergence ahead of the continuation refactor.
;; CHECK: error: define-values expected 2 values, got 3
(linklet () () (define-values (x y) (values 1 2 3)))
