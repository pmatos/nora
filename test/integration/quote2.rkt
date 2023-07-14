;; RUN: norac %s | FileCheck %s
;; CHECK: ''x
(linklet () ()
  (quote 'x))