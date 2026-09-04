;; RUN: norac %s | %normalize > %t.interp
;; RUN: %racket %s | %normalize > %t.oracle
;; RUN: diff %t.interp %t.oracle
(linklet
 ()
 ()
 (let-values (((p) (cons 1 2)))
   (begin (set-car! p 10) (set-cdr! p 20) (+ (car p) (cdr p)))))
