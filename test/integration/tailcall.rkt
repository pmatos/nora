;; RUN: norac %s | FileCheck %s
;; A deep self-tail-recursive loop runs end to end and returns its result;
;; proper tail calls keep it in bounded continuation space (see the unit tests
;; for the peak-depth assertion).
;; CHECK: 42
(linklet
 ()
 ()
 (letrec-values (((loop) (lambda (n) (if (zero? n) 42 (loop (- n 1))))))
   (loop 100000)))
