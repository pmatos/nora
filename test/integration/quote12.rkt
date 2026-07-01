;; RUN: norac %s | FileCheck %s
;; CHECK: '#(1 2)
(linklet () () '#(1 2))
