;; REQUIRES: racket
;; RUN: %observe %s | FileCheck %s
;; CHECK: 3
;;
;; The genuine hygiene hole structural normalization cannot close: `dup`'s
;; transformer binds two identifiers both named `tmp` in the *same*
;; let-values clause list — one built via (datum->syntax stx 'tmp), the
;; other via (datum->syntax (quote-syntax here) 'tmp) — so they are
;; hygienically distinct (Racket accepts the "duplicate" binding without
;; error) but print identically, since syntax->datum strips all scope-set
;; information. Confirmed empirically: %racket-expand %s | %normalize
;; produces
;;   (let-values (((v11) (quote 1)) ((v12) (quote 2))) (#%app + v12 v12))
;; — both references to `tmp` collapse onto the *second* clause's binder
;; (v12), silently losing the reference to the first one (v11) entirely.
;; This is not a bug to fix by more clever renaming: the two `tmp`
;; occurrences in the body are textually indistinguishable once
;; syntax->datum has erased their scope sets, so no purely-structural,
;; textual walk can tell them apart. Hence %observe: evaluate the module
;; directly (skipping the lossy syntax->datum round trip) and compare
;; observable output instead of normalized text.
(module m racket/base
  (require (for-syntax racket/base))
  (define-syntax (dup stx)
    (define id1 (datum->syntax stx 'tmp))
    (define id2 (datum->syntax (quote-syntax here) 'tmp))
    (with-syntax ([id1 id1] [id2 id2])
      #'(let-values (((id1) 1) ((id2) 2)) (+ id1 id2))))
  (dup))
