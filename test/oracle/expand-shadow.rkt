;; REQUIRES: racket
;; RUN: %racket-expand %s | %normalize > %t.out
;; RUN: FileCheck %s < %t.out
;; Two lexically-distinct lambdas both name their parameter `t`. Hygiene
;; has no reason to rename either (they never collide), so `expand` prints
;; both occurrences as plain `t` — but this is a *global*-consistent
;; normalizer, not local alpha-renaming, so the same source text bound in
;; two different, non-nested scopes must still resolve to two different
;; canonical names, not collide into one.
;; CHECK: (lambda (v2) v2)
;; CHECK: (lambda (v3) v3)
(module m racket/base
  (define-values (f) (lambda (t) t))
  (define-values (g) (lambda (t) t)))
