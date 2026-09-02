;; RUN: norac %s | FileCheck %s
;; (f 0) is a tail call of the enclosing with-continuation-mark, which is
;; itself in tail position of the linklet body, so f's activation reuses the
;; same continuation frame as the outer mark: installing 'k again replaces
;; the outer value (1) rather than stacking alongside it.
;; CHECK: (2)
(linklet () ()
  (define-values (f)
    (lambda (x)
      (with-continuation-mark 'k 2
        (continuation-mark-set->list (current-continuation-marks) 'k))))
  (with-continuation-mark 'k 1 (f 0)))
