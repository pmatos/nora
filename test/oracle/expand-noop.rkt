;; REQUIRES: racket
;; RUN: %racket-expand %s | %normalize > %t.run1
;; RUN: %racket-expand %s | %normalize > %t.run2
;; RUN: diff %t.run1 %t.run2
(module m racket/base (define-values (x) 1) x)
