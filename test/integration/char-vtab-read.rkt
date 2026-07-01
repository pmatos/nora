;; RUN: norac %s | FileCheck %s
;; CHECK: #\vtab
(linklet () () '#\vtab)
