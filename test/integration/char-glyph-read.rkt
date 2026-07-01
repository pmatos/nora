;; RUN: norac %s | FileCheck %s
;; Reading a printed multi-byte glyph char literal back must yield the same char.
;; CHECK: #\λ
(linklet () () '#\λ)
