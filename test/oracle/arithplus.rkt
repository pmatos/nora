;; RUN: norac %s | %normalize > %t.interp
;; RUN: %racket %s | %normalize > %t.oracle
;; RUN: diff %t.interp %t.oracle
(linklet () () (+ 2 0))
