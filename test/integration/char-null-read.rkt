;; RUN: norac %s | FileCheck %s
;; The traditional #\null name must still lex (not be truncated to #\nul), but
;; it canonicalizes to #\nul on write, matching real Racket
;; (`racket -e "(write '#\null)"` prints #\nul) and issue #73's finding that
;; nul/vtab/page/rubout are Racket's correct print names.
;; CHECK: #\nul
(linklet () () '#\null)
