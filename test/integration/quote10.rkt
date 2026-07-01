;; RUN: norac %s | FileCheck %s
;; CHECK: "z"
(linklet () () '"z")
