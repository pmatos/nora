;; RUN: norac %s | FileCheck %s
;; CHECK: #<variable-reference>
(linklet () () (#%variable-reference))
