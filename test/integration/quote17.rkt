;; RUN: norac %s | FileCheck %s
;; CHECK: #\space
(linklet () () '#\u0020)
