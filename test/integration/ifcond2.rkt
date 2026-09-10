;; RUN: norac %s | FileCheck %s
;; CHECK: 2
(linklet () ()
  (if #f 1 2))
