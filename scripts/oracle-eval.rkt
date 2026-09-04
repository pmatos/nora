#lang racket/base
;; oracle-eval.rkt — the %racket driver for test/oracle/*.rkt fixtures.
;;
;; Reads a fixture's single `(linklet () () form ...)` datum, rewrites it in
;; memory (the file on disk is never touched) into a linklet that exports the
;; value of its last top-level form, compiles + instantiates it via
;; racket/linklet, then `write`s that value followed by a newline — the same
;; shape norac's main.cpp produces (`I.getResult()->write()` + '\n'), so the
;; two engines' output can be diffed directly after %normalize.
;;
;; Assumes the fixture's last top-level form is an expression, never a
;; define-values — true of every fixture under test/oracle/ and
;; test/integration/ today.

(require racket/linklet)

(define (read-linklet-datum path)
  (call-with-input-file path
    (lambda (in)
      (define datum (read in))
      (unless (eof-object? (read in))
        (error 'oracle-eval "expected a single datum in ~a" path))
      datum)))

;; Splits a non-empty list into (values all-but-last last).
(define (split-last forms)
  (let loop ([forms forms] [init '()])
    (if (null? (cdr forms))
        (values (reverse init) (car forms))
        (loop (cdr forms) (cons (car forms) init)))))

(define (rewrite-linklet datum)
  (unless (and (pair? datum) (eq? (car datum) 'linklet) (>= (length datum) 4))
    (error 'oracle-eval "expected a (linklet imports exports form ...) form, got ~a" datum))
  (define imports (list-ref datum 1))
  (define body (cdddr datum))
  (when (null? body)
    (error 'oracle-eval "linklet body is empty"))
  (define-values (init-forms last-form) (split-last body))
  `(linklet ,imports (oracle-result)
     ,@init-forms
     (define-values (oracle-result) ,last-form)))

(module+ main
  (define args (current-command-line-arguments))
  (unless (= (vector-length args) 1)
    (error 'oracle-eval "usage: oracle-eval.rkt <fixture.rkt>"))
  (define datum (read-linklet-datum (vector-ref args 0)))
  (define compiled (compile-linklet (rewrite-linklet datum)))
  (define instance (instantiate-linklet compiled '()))
  (write (instance-variable-value instance 'oracle-result))
  (newline))
