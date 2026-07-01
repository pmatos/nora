;; RUN: norac %s | FileCheck %s
;; The traditional #\null name must still lex (not be truncated to #\nul).
;; CHECK: #\null
(linklet () () '#\null)
