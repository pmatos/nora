#!/usr/bin/env python3
import sys
import subprocess


# Emulate the `not` tool from LLVM's test infrastructure for use with lit and
# FileCheck. It succeeds if the given subcommand fails and vice versa.
def main():
    cmd = sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(0 if result.returncode != 0 else 1)


if __name__ == '__main__':
    main()
