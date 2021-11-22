#lang racket/base
;; Initial compiler / expander, mostly ripped from several places in Racketscript 

(require (for-syntax racket/base)
         racket/format
         racket/path
         racket/pretty
         racket/runtime-path
         racket/set
         syntax/parse
         syntax/stx
         "logging.rkt")
       
(define-runtime-path nora-main-module "main.rkt")

(define nora-compiler-dir (path-only nora-main-module))

(define nora-runtime-dir
  (build-path nora-compiler-dir "runtime"))

(define primitive-modules
  (set '#%runtime
       '#%core
       '#%main
       '#%read
       '#%kernel
       '#%paramz
       '#%unsafe
       '#%utils
       '#%flfxnum
       '#%futures
       '#%extfl
       '#%place-struct
       '#%network
       '#%builtin
       '#%boot
       '#%foreign
       '#%place
       '#%linklet-primitive
       (build-path nora-runtime-dir "lib.rkt")))

(define (primitive-module? mod)
  (set-member? primitive-modules mod))

(define (actual-module-path in-path)
  (cond
    [(path? in-path) in-path]
    [(and (symbol? in-path) (primitive-module? in-path))
     (build-path nora-runtime-dir
                 (~a (substring (symbol->string in-path) 2) ".rkt"))]
    [else (error 'actual-module-path "~a is not a primitive module" in-path)]))

(define (read-module input)
  (read-syntax (object-name input) input))

(define (open-read-module in-path)
  (call-with-input-file (actual-module-path in-path)
    (lambda (in)
      (read-module in))))

(define (do-expand stx)
  ;; error checking
  (syntax-parse stx
    [((~and mod-datum (~datum module)) n:id lang:expr . rest)
     (void)]
    [((~and mod-datum (~datum module)) . rest)
     (error 'do-expand
            "got ill-formed module: ~a\n" (syntax->datum #'rest))]
    [rest
     (error 'do-expand
            "got something that isn't a module: ~a\n" (syntax->datum #'rest))])
  ;; work

  (parameterize ([current-namespace (make-base-namespace)])
    (expand stx)))


(define (quick-expand in-path)
  (log-nora-info "[expand] ~a" in-path)
  (read-accept-reader #true)
  (read-accept-lang #true)
  (define full-path (path->complete-path (actual-module-path in-path)))
  (parameterize ([current-directory (path-only full-path)])
    (do-expand (open-read-module (file-name-from-path in-path)))))

(module+ main
  (require racket/cmdline)

  (define source
    (command-line
     #:program "nora"
     #:args (filename)
     (string->path filename)))
  
  (define expanded-module (quick-expand source))
  (pretty-write (syntax->datum expanded-module)))
      
