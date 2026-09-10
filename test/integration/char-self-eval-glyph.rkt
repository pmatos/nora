;; RUN: norac %s | FileCheck %s
;; A non-ASCII glyph character is also self-evaluating.
;; CHECK: #\λ
(linklet () () #\λ)
