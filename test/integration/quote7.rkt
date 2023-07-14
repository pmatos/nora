;; RUN: norac %s | FileCheck %s
;; CHECK: '(you can 'me)
(linklet () () '(you can 'me))