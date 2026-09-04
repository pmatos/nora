#lang racket/base
;; oracle-observe.rkt — the %observe driver: an observational-equivalence
;; fallback for fixtures where structural normalization provably cannot
;; reconcile two hygienically-distinct, same-printed-name binders (see
;; test/oracle/expand-shadow-hygiene.rkt for the motivating case).
;;
;; Deliberately does not go through %racket-expand/syntax->datum at all:
;; that round-trip is exactly the lossy step that erases the scope
;; information distinguishing two same-named-but-distinct binders in the
;; first place (confirmed empirically — re-reading and re-evaluating such
;; a fixture's syntax->datum text fails outright with a "duplicate
;; binding name" error, since plain `read` produces fresh, unhygienic
;; symbols). Instead this evaluates the fixture's original module datum
;; directly and prints whatever the module prints when instantiated.

(define (read-datum path)
  (call-with-input-file path
    (lambda (in)
      (define datum (read in))
      (unless (eof-object? (read in))
        (error 'oracle-observe "expected a single datum in ~a" path))
      datum)))

(module+ main
  (define args (current-command-line-arguments))
  (unless (= (vector-length args) 1)
    (error 'oracle-observe "usage: oracle-observe.rkt <fixture.rkt>"))
  (define datum (read-datum (vector-ref args 0)))
  (parameterize ([current-namespace (make-base-namespace)])
    (eval datum)
    (dynamic-require ''m #f)))
