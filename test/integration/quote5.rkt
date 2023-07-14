;; RUN: norac %s | FileCheck %s
;; CHECK: #\a
(linklet () ()
  '#\a)