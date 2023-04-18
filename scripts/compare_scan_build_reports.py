import sys
import os
from bs4 import BeautifulSoup


def parse_reports(report_dir):
    error_set = set()
    for root, dirs, files in os.walk(report_dir):
        for file in files:
            if file.endswith(".html"):
                with open(os.path.join(root, file), "r") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    for row in soup.find_all("tr"):
                        error_set.add("".join(row.stripped_strings))
    return error_set


if __name__ == "__main__":
    base_analysis_dir = sys.argv[1]
    pr_analysis_dir = sys.argv[2]

    base_errors = parse_reports(base_analysis_dir)
    pr_errors = parse_reports(pr_analysis_dir)
    new_errors = pr_errors - base_errors

    if new_errors:
        print(f"New errors found: {len(new_errors)}")
        sys.exit(1)
