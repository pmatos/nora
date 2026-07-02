;; RUN: norac %s | FileCheck %s
;; A multi-byte glyph char followed by whitespace (inside a list) must lex too;
;; the closing-delimiter case is covered by char-glyph-read.rkt.
;; CHECK: '(#\λ #\a)
(linklet () () '(#\λ #\a))
