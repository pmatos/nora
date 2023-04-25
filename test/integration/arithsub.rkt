;; RUN: norac %s | FileCheck %s
;; CHECK: 1
(linklet () ()
  (let-values (((x y) (values -1 0)))
    (- y x)
  )
)