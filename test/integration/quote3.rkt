;; RUN: norac %s | FileCheck %s
;; CHECK: '(define foo 2)
(linklet () ()
  '(define foo 2))