#!/usr/bin/env python3
"""%normalize — normalizes one engine's printed output before oracle diffing.

For M0 the oracle corpus is scalar-only (see test/oracle/), so this only
strips trailing whitespace per line and enforces exactly one trailing
newline. M0-N's alpha/gensym normalizer (issue #92) replaces this
implementation behind the same %normalize substitution.
"""
import sys


def main():
    data = sys.stdin.read()
    lines = [line.rstrip() for line in data.split('\n')]
    sys.stdout.write('\n'.join(lines).rstrip('\n') + '\n')


if __name__ == '__main__':
    main()
