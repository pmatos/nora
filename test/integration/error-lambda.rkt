;; RUN: not norac %s 2>&1 | FileCheck %s
;; A lambda missing its formals is a parse-time error at the commit point.
;; CHECK: error: expected formals after 'lambda'
(linklet () () (lambda))
