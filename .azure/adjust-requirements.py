"""Simple package switch."""

import re
import sys


def main(pkg: str, req_file: str = "requirements.txt"):
    """Perform the replacement."""
    with open(req_file) as fo:
        lines = fo.readlines()
    lines = [re.sub(r"lightning([ <=>]+)", rf"{pkg} \1", ln) for ln in lines]
    with open(req_file, "w") as fw:
        fw.writelines(lines)


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    main(sys.argv[1])
