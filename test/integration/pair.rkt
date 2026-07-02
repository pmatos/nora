;; RUN: norac %s | FileCheck %s
;; A pair is a shared mutable cons cell: set-car!/set-cdr! through one reference
;; to `p` are visible when its fields are read again (M2's shared value model).
;; CHECK: 30
(linklet
 ()
 ()
 (let-values (((p) (cons 1 2)))
   (begin (set-car! p 10) (set-cdr! p 20) (+ (car p) (cdr p)))))
