#lang racket/base
;; oracle-expand.rkt — the %racket-expand driver for test/oracle/*.rkt
;; fixtures that exercise the M0-N alpha/gensym normalizer (issue #92).
;;
;; Reads a fixture's single module datum, expands it in a fresh namespace
;; (confirmed empirically: a fresh namespace per process reproduces the same
;; gensym-counter baseline across separate process invocations of a given
;; fixture), and `write`s `(syntax->datum expanded)` followed by a newline.
;;
;; An optional second CLI argument, a warmup count N, calls `(gensym)` N
;; times before `expand` runs, shifting the process's gensym-counter
;; baseline by exactly N — the controlled perturbation
;; test/oracle/expand-gensym-shift.rkt uses to prove the normalizer
;; tolerates a shifted counter.

(define (read-datum path)
  (call-with-input-file path
    (lambda (in)
      (define datum (read in))
      (unless (eof-object? (read in))
        (error 'oracle-expand "expected a single datum in ~a" path))
      datum)))

(module+ main
  (define args (current-command-line-arguments))
  (unless (member (vector-length args) '(1 2))
    (error 'oracle-expand "usage: oracle-expand.rkt <fixture.rkt> [warmup-count]"))
  (define datum (read-datum (vector-ref args 0)))
  (define warmup (if (= (vector-length args) 2)
                      (string->number (vector-ref args 1))
                      0))
  (for ([_ (in-range warmup)]) (gensym))
  (parameterize ([current-namespace (make-base-namespace)])
    (write (syntax->datum (expand datum)))
    (newline)))
