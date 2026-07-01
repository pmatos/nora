;; RUN: not norac %s 2>&1 | FileCheck %s
;; A top-level form that is not a linklet is reported at the offending token.
;; CHECK: error: expected 'linklet' keyword
(foo bar)
