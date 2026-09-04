#!/usr/bin/env python3
"""%normalize — normalizes one engine's printed output before oracle diffing.

For M0's scalar-only fixtures (`2`, `(1 2 3)`, ...) this only strips
trailing whitespace per line and enforces exactly one trailing newline, as
before. For a fully-expanded `(module ...)` datum — M0-N's alpha/gensym
normalizer, issue #92 — it renames every bound identifier and renumbers
every quoted gensym to a stable canonical form instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oracle_alpha
import oracle_datum


def _is_module_datum(datum):
    return (isinstance(datum, list) and datum
            and isinstance(datum[0], oracle_datum.Symbol)
            and datum[0].name == 'module')


def _normalize_whitespace(text):
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).rstrip('\n') + '\n'


def main():
    data = sys.stdin.read()
    datum = oracle_datum.parse(data)
    if _is_module_datum(datum):
        sys.stdout.write(oracle_datum.write(oracle_alpha.normalize(datum)) + '\n')
    else:
        sys.stdout.write(_normalize_whitespace(data))


if __name__ == '__main__':
    main()
