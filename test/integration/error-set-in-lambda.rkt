;; RUN: not norac %s 2>&1 | FileCheck %s
;; A set! on an unbound identifier inside a lambda body (which is cloned into the
;; closure when the lambda is evaluated) must still report the identifier's
;; source location, not <unknown>:0 — i.e. the range survives cloning.
;; CHECK: {{[0-9]+}}:{{[0-9]+}}: error: cannot set unbound identifier: x
;; CHECK-NEXT: {{.*}}set! x
;; CHECK-NEXT: {{\^}}
(linklet () () ((lambda () (set! x 5))))
