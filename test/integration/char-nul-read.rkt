;; RUN: norac %s | FileCheck %s
;; CHECK: #\nul
(linklet () () '#\nul)
