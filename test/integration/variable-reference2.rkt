;; RUN: norac %s | FileCheck %s
;; CHECK: #<variable-reference>
(linklet () ()
  (define-values (x) (values 1))
  (#%variable-reference (#%top . x)))
