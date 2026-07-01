;; RUN: norac %s | FileCheck %s
;; with-continuation-mark evaluates key and value, then returns the value of
;; its result expression (in tail position).
;; CHECK: 5
(linklet () ()
  (with-continuation-mark 'k 1 (+ 2 3)))
