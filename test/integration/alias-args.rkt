;; RUN: norac %s | FileCheck %s
;; (f x x) passes one shared box as both arguments of a two-argument lambda:
;; two separate environment lookups of the same identifier feed two different
;; argument slots, but a mutation through one parameter is visible through the
;; other (M2/GC S3's own example: the environment shares instead of cloning on
;; lookup).
;; CHECK: 10
(linklet
 ()
 ()
 (define-values (f)
   (lambda (a b) (begin (set-box! a 10) (unbox b))))
 (let-values (((x) (box 1)))
   (f x x)))
