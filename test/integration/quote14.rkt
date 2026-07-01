;; RUN: norac %s | FileCheck %s
;; CHECK: #\space
(linklet () () '#\space)
