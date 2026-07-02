;; RUN: not norac %s 2>&1 | FileCheck %s
;; An unbound identifier is reported with a source location and a caret.
;; CHECK: {{[0-9]+}}:{{[0-9]+}}: error: unbound identifier: foo
;; CHECK-NEXT: {{.*}}foo
;; CHECK-NEXT: {{\^}}
(linklet () () foo)
