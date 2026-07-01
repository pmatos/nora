;; RUN: norac %s | FileCheck %s
;; XFAIL: *
;; TODO: quoting characters needs a Char value node (not implemented yet).
;; CHECK: #\a
(linklet () ()
  '#\a)