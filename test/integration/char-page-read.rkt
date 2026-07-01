;; RUN: norac %s | FileCheck %s
;; CHECK: #\page
(linklet () () '#\page)
