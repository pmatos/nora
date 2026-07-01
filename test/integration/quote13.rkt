;; RUN: norac %s | FileCheck %s
;; CHECK: '#("z" x)
(linklet () () '#("z" x))
