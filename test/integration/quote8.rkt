;; RUN: norac %s | FileCheck %s
;; XFAIL: *
;; TODO: needs string and vector value nodes plus improper-list (dotted tail)
;; printing; none are implemented yet.
;; CHECK: '((1 2 3) #("z" x) . the-end)
(linklet () () (quote ((1 2 3) #("z" x) . the-end)))