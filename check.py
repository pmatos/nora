import sys
import os
import subprocess
from collections import OrderedDict

nora_root = os.getcwd()
nora_bin = os.path.join(nora_root, 'bin')

def run_lit():
    global num_failures
    num_failures = 0    
    lit_script = os.path.join(nora_bin, 'nora-lit')
    lit_tests = os.path.join(nora_root, 'test', 'lit')
    # lit expects to be run as its own executable
    cmd = [sys.executable, lit_script, lit_tests, '-vv']
    result = subprocess.run(cmd)
    if result.returncode != 0:
        num_failures += 1

TEST_SUITES = OrderedDict([
    ('lit', run_lit),
])

# Run all the tests
def main():
    all_suites = TEST_SUITES.keys()

    for test in all_suites:
        TEST_SUITES[test]()

    # Check/display the results
    if num_failures == 0:
        print('\n[ success! ]')

    if num_failures > 0:
        print('\n[ ' + str(num_failures) + ' failures! ]')
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
