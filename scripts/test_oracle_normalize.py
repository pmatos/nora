#!/usr/bin/env python3
"""Unit tests for the oracle-harness normalizer (issue #92).

stdlib unittest, not pytest: CI's oracle job only `pip install`s `lit`, and
this suite must run without a `racket` binary too, so it belongs in the
default `ctest` matrix (see test/oracle/CMakeLists.txt) rather than the
opt-in oracle job.
"""
import unittest

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


if __name__ == '__main__':
    unittest.main()
