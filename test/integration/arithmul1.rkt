;; RUN: norac %s | FileCheck %s
;; CHECK: -20
(linklet () () (* 2 -10))
