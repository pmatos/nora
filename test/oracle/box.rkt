;; REQUIRES: racket
;; RUN: norac %s | %normalize > %t.interp
;; RUN: %racket %s | %normalize > %t.oracle
;; RUN: diff %t.interp %t.oracle
(linklet
 ()
 ()
 (let-values (((b) (box 1)))
   (begin (set-box! b 10) (unbox b))))
