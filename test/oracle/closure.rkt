;; REQUIRES: racket
;; RUN: norac %s | %normalize > %t.interp
;; RUN: %racket %s | %normalize > %t.oracle
;; RUN: diff %t.interp %t.oracle
(linklet () ()
  (define-values (fn) (values 0))
  (let-values (((x) (values 2)))
     (set! fn (lambda (y) (+ x y))))
  (let-values (((x) (values 3)))
     (fn 10)))
