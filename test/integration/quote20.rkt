;; RUN: norac %s | FileCheck %s
;; CHECK: '(#:foo 1)
(linklet () () '(#:foo 1))
