;; RUN: norac %s | FileCheck %s
;; continuation-mark-set-first returns #f when no mark matches the key.
;; CHECK: #f
(linklet () ()
  (with-continuation-mark 'k 1
    (continuation-mark-set-first (current-continuation-marks) 'missing)))
