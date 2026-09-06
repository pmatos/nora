;; RUN: norac %s | FileCheck %s
;; (+ x x), (- x x), and (* x x) each look up the same identifier twice, once
;; per operand. Pins that the arithmetic accumulators (AddFunction/
;; SubtractFunction/MultiplyFunction) are unaffected now that both lookups
;; alias the same materialized value instead of being independent clones
;; (M2/GC S3).
;; CHECK: 15
(linklet
 ()
 ()
 (let-values (((x) 5))
   (- (* x x) (+ x x))))
