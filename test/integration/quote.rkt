;; RUN: norac %s | FileCheck %s
;; CHECK: 'x
(linklet () ()
  'x)