;; REQUIRES: racket
;; RUN: %racket-expand %s | %normalize > %t.plain
;; RUN: %racket-expand %s 7 | %normalize > %t.warmup7
;; RUN: diff %t.plain %t.warmup7
(module m racket/base
  (require (for-syntax racket/base))
  (define-syntax (mymac stx)
    (with-syntax ([s (gensym)])
      #''s))
  (mymac))
