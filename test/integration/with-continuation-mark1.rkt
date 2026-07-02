;; RUN: norac %s | FileCheck %s
;; The installed mark is visible while the result expression runs and can be
;; read back with continuation-mark-set-first on the current marks.
;; CHECK: 1
(linklet () ()
  (with-continuation-mark 'k 1
    (continuation-mark-set-first (current-continuation-marks) 'k)))
