;; RUN: norac %s | FileCheck %s
;; CHECK: '(1 2 3)
(linklet () () (quote (1 2 . (3))))