;; RUN: norac %s | FileCheck %s
;; gensym returns a fresh uninterned symbol, never eq? to any other, so eq? on
;; symbols is object identity (interned symbols with the same name are eq?; a
;; gensym is not) — M2's shared value model.
;; CHECK: #f
(linklet () () (eq? (gensym) (gensym)))
