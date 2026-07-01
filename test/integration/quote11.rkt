;; RUN: norac %s | FileCheck %s
;; CHECK: '(1 2 . the-end)
(linklet () () '(1 2 . the-end))
