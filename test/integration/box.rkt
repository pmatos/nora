;; RUN: norac %s | FileCheck %s
;; A box is a shared mutable cell: set-box! through one reference to `b` is
;; visible when `b` is read again (M2's shared value model).
;; CHECK: 10
(linklet
 ()
 ()
 (let-values (((b) (box 1)))
   (begin (set-box! b 10) (unbox b))))
