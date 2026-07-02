;; RUN: norac %s | FileCheck %s
;; CHECK: 6
;; A multiple-values clause binds several identifiers at once, and a later
;; clause can refer to them.
(linklet () ()
  (letrec-values ([(a b) (values 1 2)]
                  [(c) (+ a b)])
    (+ c (+ a b))))
