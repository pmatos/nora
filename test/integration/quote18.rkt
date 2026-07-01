;; RUN: norac %s | FileCheck %s
;; CHECK: #\newline
(linklet () () '#\u000a)
