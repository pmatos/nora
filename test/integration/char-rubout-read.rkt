;; RUN: norac %s | FileCheck %s
;; CHECK: #\rubout
(linklet () () '#\rubout)
