;; RUN: norac %s | FileCheck %s
;; CHECK: 1
(linklet () ()
  (let-values (((x) (values -1)))
    (- x)
  )
)