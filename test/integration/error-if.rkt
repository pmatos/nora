;; RUN: not norac %s 2>&1 | FileCheck %s
;; An if form missing its else branch is a parse error at the commit point.
;; CHECK: error: expected else expression in 'if'
(linklet () () (if #t 1))
