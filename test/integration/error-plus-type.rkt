;; RUN: not norac %s 2>&1 | FileCheck %s
;; A runtime type error surfaces at the call site.
;; CHECK: error: invalid arguments to '+'
(linklet () () (+ 1 #t))
