;; RUN: norac %s | FileCheck %s
;; CHECK: '#:foo
(linklet () () '#:foo)
