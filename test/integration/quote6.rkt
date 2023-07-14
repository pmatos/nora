;; RUN: norac %s | FileCheck %s
;; CHECK: '(linklet () () (define-values 2 3) (kaboom broken syntax))
(linklet () ()
  '(linklet () () (define-values 2 3) (kaboom broken syntax)))