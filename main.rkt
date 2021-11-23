#lang racket/base

(require "compiler.rkt"
         racket/pretty)


(module+ main
  (require racket/cmdline)

  (define source
    (command-line
     #:program "nora"
     #:args (filename)
     (string->path filename)))
  
  (define expanded-module (quick-expand source))
  (pretty-write (syntax->datum expanded-module)))
      
