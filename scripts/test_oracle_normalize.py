#!/usr/bin/env python3
"""Unit tests for the oracle-harness normalizer (issue #92).

stdlib unittest, not pytest: CI's oracle job only `pip install`s `lit`, and
this suite must run without a `racket` binary too, so it belongs in the
default `ctest` matrix (see test/oracle/CMakeLists.txt) rather than the
opt-in oracle job.
"""
import unittest

import oracle_alpha
import oracle_datum


class DatumRoundTrip(unittest.TestCase):
    def check(self, text):
        datum = oracle_datum.parse(text)
        self.assertEqual(oracle_datum.write(datum), text)

    def test_dotted_list(self):
        self.check('(1 2 . the-end)')

    def test_quote_form(self):
        self.check('(quote foo)')

    def test_boolean_true(self):
        self.check('#t')

    def test_boolean_false(self):
        self.check('#f')

    def test_string(self):
        self.check('"a str"')

    def test_char(self):
        self.check('#\\a')

    def test_vector(self):
        self.check('#(1 2 3)')

    def test_define_values(self):
        self.check('(define-values (x) 1)')

    def test_hash_percent_symbol(self):
        self.check('(#%app + x y)')

    def test_empty_list(self):
        self.check('()')

    def test_negative_number(self):
        self.check('(- 1 2)')


class AlphaRenaming(unittest.TestCase):
    def normalized(self, text):
        return oracle_datum.write(oracle_alpha.normalize(oracle_datum.parse(text)))

    def assert_normalize_equal(self, text_a, text_b):
        self.assertEqual(self.normalized(text_a), self.normalized(text_b))

    def test_define_values_name_is_renamed(self):
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin '
            '(define-values (lifted/15) 1) lifted/15))',
            '(module m racket/base (#%module-begin '
            '(define-values (lifted/9) 1) lifted/9))',
        )

    def test_lambda_parameter_is_renamed(self):
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (y) (#%app + y y)))))',
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (z) (#%app + z z)))))',
        )

    def test_if_and_app_recurse_without_renaming_globals(self):
        text = ('(module m racket/base (#%module-begin '
                '(define-values (f) (lambda (y) (if y (#%app + y 1) 0)))))')
        out = self.normalized(text)
        self.assertIn('+', out)
        self.assertIn('if', out)

    def test_module_body_forward_reference_uses_same_name(self):
        # `a`'s RHS refers to `b`, defined afterwards; both occurrences of
        # `b`'s canonical name must match since #%module-begin scopes its
        # whole body at once, not left-to-right.
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(define-values (a) (if b 1 2)) (define-values (b) 3)))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(define-values (v0) (if v1 1 2)) (define-values (v1) 3)))')

    def test_submodule_does_not_inherit_outer_bindings(self):
        # A nested module never sees its enclosing module's bindings, so a
        # same-named free reference inside it stays unbound (unrenamed).
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(define-values (x) 1) '
            '(module sub racket/base (#%module-begin x))))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(define-values (v0) 1) '
            '(module sub racket/base (#%module-begin x))))')

    def test_define_syntaxes_is_an_isolated_phase_1_scope(self):
        # A phase-0 define-values named `x` must not leak into a
        # define-syntaxes transformer body that also happens to use `x`.
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(define-values (x) 1) '
            '(define-syntaxes (mymac) (lambda (x) x))))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(define-values (v0) 1) '
            '(define-syntaxes (mymac) (lambda (v1) v1))))')


class LetValuesRenaming(unittest.TestCase):
    def normalized(self, text):
        return oracle_datum.write(oracle_alpha.normalize(oracle_datum.parse(text)))

    def assert_normalize_equal(self, text_a, text_b):
        self.assertEqual(self.normalized(text_a), self.normalized(text_b))

    def test_let_values_temp_name_is_renamed(self):
        # The let*/lift-desugaring shape: box.rkt's temp is named `b` here,
        # `tmp` in the other variant — both must normalize identically.
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (r) '
            '(let-values (((b) (box 1))) (begin (set-box! b 10) '
            '(unbox b))))))',
            '(module m racket/base (#%module-begin (define-values (r) '
            '(let-values (((tmp) (box 1))) (begin (set-box! tmp 10) '
            '(unbox tmp))))))',
        )

    def test_let_values_clause_does_not_see_its_own_binding(self):
        # rhs of a let-values clause is evaluated in the *outer* env, so a
        # same-named outer free variable in the rhs must stay unrenamed by
        # the clause's own (unrelated) binder.
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(let-values (((x) x)) x)))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(let-values (((v0) x)) v0)))')

    def test_letrec_values_clauses_forward_reference(self):
        # letrec-values clauses may reference each other regardless of
        # order — both variants name the two locals differently but must
        # normalize identically. Applications use #%app, matching real
        # fully-expanded shape (a bare `(odd? n)` would look like an
        # unknown special form to the generic fallback, not a call).
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (r) '
            '(letrec-values (((even?) (lambda (n) (if n (#%app odd? n) #t))) '
            '((odd?) (lambda (n) (if n (#%app even? n) #f)))) even?))))',
            '(module m racket/base (#%module-begin (define-values (r) '
            '(letrec-values (((e) (lambda (n) (if n (#%app o n) #t))) '
            '((o) (lambda (n) (if n (#%app e n) #f)))) e))))',
        )


class RemainingGrammarRenaming(unittest.TestCase):
    """Covers case-lambda (its own binder form) plus set!/begin/begin0/
    with-continuation-mark/#%variable-reference, all of which have no
    binding power and are already handled by _walk's generic fallback —
    these are regression tests, not new production code.

    #%top and #%expression are deliberately not exercised here: confirmed
    empirically that `expand` always eliminates both by the time a program
    reaches syntax->datum (`(#%top . x)` becomes the bare resolved `x`;
    `(#%expression e)` becomes just `e`), so neither has a reachable
    representation for this normalizer to ever see.
    """

    def normalized(self, text):
        return oracle_datum.write(oracle_alpha.normalize(oracle_datum.parse(text)))

    def assert_normalize_equal(self, text_a, text_b):
        self.assertEqual(self.normalized(text_a), self.normalized(text_b))

    def test_case_lambda_clauses_have_independent_scopes(self):
        # Proper, dotted, and bare-symbol formals in the same case-lambda;
        # differently-named in each variant but must normalize identically.
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (f) '
            '(case-lambda ((a) a) ((a b . rest) rest) (args args)))))',
            '(module m racket/base (#%module-begin (define-values (f) '
            '(case-lambda ((x) x) ((x y . z) z) (w w)))))',
        )

    def test_set_bang_renames_target_as_a_reference_not_a_binder(self):
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (x) (set! x (#%app + x 1)) x))))',
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (y) (set! y (#%app + y 1)) y))))',
        )

    def test_begin_begin0_and_with_continuation_mark_recurse(self):
        self.assert_normalize_equal(
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (x) (if x (begin (set! x 2) x) 3) '
            '(begin0 x (set! x 3)) '
            '(with-continuation-mark (quote k) (quote v) '
            '(#%variable-reference x))))))',
            '(module m racket/base (#%module-begin (define-values (f) '
            '(lambda (y) (if y (begin (set! y 2) y) 3) '
            '(begin0 y (set! y 3)) '
            '(with-continuation-mark (quote k) (quote v) '
            '(#%variable-reference y))))))',
        )


class ProvideRequireBoundary(unittest.TestCase):
    def normalized(self, text):
        return oracle_datum.write(oracle_alpha.normalize(oracle_datum.parse(text)))

    def test_provided_name_keeps_its_text_definition_and_clause(self):
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(#%provide exported) '
            '(define-values (exported) 1) '
            '(define-values (internal) 2)))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(#%provide exported) '
            '(define-values (exported) 1) '
            '(define-values (v0) 2)))')

    def test_provided_rename_clause_keeps_internal_name(self):
        # (#%provide (rename internal external)): `internal` is the
        # define-values LHS being exported (kept verbatim); `external` is
        # just a public label, never a binder, so it always passes through.
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(#%provide (rename internal external)) '
            '(define-values (internal) 1)))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(#%provide (rename internal external)) '
            '(define-values (internal) 1)))')

    def test_require_module_path_passes_through_untouched(self):
        out = self.normalized(
            '(module m racket/base (#%module-begin '
            '(#%require racket/list (for-syntax racket/base))))')
        self.assertEqual(
            out,
            '(module m racket/base (#%module-begin '
            '(#%require racket/list (for-syntax racket/base))))')


if __name__ == '__main__':
    unittest.main()
