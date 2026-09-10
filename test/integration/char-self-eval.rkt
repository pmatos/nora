;; RUN: norac %s | FileCheck %s
;; Characters are self-evaluating, exactly like booleans/integers/strings: a
;; bare (unquoted) char literal reads and evaluates directly.
;; CHECK: #\a
(linklet () () #\a)
