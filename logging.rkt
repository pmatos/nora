#lang racket/base

(require (for-syntax racket/base
                     racket/syntax
                     syntax/parse)
         (for-meta 2 syntax/parse))

(provide log-nora-info
         log-nora-debug
         log-nora-warning
         log-nora-error)

(define logging? (make-parameter #true))

(define-syntax log-nora
  (syntax-parser
    [(_ kind:id)
     #:with name (format-id #'kind "log-nora-~a" (syntax-e #'kind) #:source #'kind)
     #:with str (symbol->string (syntax-e #'kind))
     (syntax (define-syntax name
               (syntax-parser
                 [(_ a0 a* (... ...))
                  #'(when (logging?)
                      (begin (printf "[~a]" 'str)
                             (unless (equal? (string-ref a0 0) #\[)
                               (printf " "))
                             (printf a0 a* (... ...))
                             (printf "\n")))])))]))
(log-nora info)
(log-nora error)
(log-nora warning)
(log-nora debug)
