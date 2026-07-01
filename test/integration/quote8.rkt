;; RUN: norac %s | FileCheck %s
;; CHECK: '((1 2 3) #("z" x) . the-end)
(linklet () () (quote ((1 2 3) #("z" x) . the-end)))