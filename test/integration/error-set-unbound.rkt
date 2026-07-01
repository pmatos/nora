;; RUN: not norac %s 2>&1 | FileCheck %s
;; set! on an unbound identifier is an error at the identifier's location.
;; CHECK: error: cannot set unbound identifier: x
(linklet () () (set! x 5))
