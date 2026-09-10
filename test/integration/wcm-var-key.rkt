;; RUN: norac %s | FileCheck %s
;; with-continuation-mark's key and value expressions are bound identifiers,
;; not literals, confirming the mark captures the looked-up value correctly
;; now that Environment shares values by reference (M2/GC S3) instead of
;; cloning them on lookup.
;; CHECK: 42
(linklet
 ()
 ()
 (let-values (((k) 'key) ((v) 42))
   (with-continuation-mark k v
     (continuation-mark-set-first (current-continuation-marks) k))))
