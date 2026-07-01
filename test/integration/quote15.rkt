;; RUN: norac %s | FileCheck %s
;; CHECK: #\λ
(linklet () () '#\u03bb)
