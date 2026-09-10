;; RUN: norac %s | FileCheck %s
;; A named character (not just a plain glyph) is also self-evaluating.
;; CHECK: #\space
(linklet () () #\space)
