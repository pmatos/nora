;; RUN: not norac %s 2>&1 | FileCheck %s
;; A free variable that is unbound where the closure is defined must NOT resolve
;; to a same-named binding at the call site: closures are lexically scoped, so
;; `y` (free and unbound in g) is an error even though the caller h binds y.
;; CHECK: Undefined Identifier: y
(linklet () ()
  (define-values (g) (lambda (x) y))
  (define-values (h) (lambda (y) (g 0)))
  (h 42))
