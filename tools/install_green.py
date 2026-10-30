"""Install the GREEN network voter (Paillard et al., 2025; Roche, Apache-2.0) where the
ensemble expects it: <repository>/neuro_green, at the exact commit used in the paper.

    python tools/install_green.py

GREEN is not on PyPI and its research code imports itself as a top-level package `green`.
The ensemble imports it as `neuro_green.green`, so this rewrites those imports and nothing
else; the network definition (`get_green`) is untouched. GREEN also needs geotorch and
lightning: pip install -e .[deep]
"""

import io
import os
import re
import shutil
import sys
import urllib.request
import zipfile

COMMIT = "b9c2a90b8ec62127ab264f44a74f3c91bd4933dd"
URL = f"https://github.com/Roche/neuro-green/archive/{COMMIT}.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "neuro_green")
REWRITES = {
    os.path.join("green", "research_code", "pl_utils.py"): [
        (r"^from green\.", "from neuro_green.green.")
    ],
    os.path.join("green", "research_code", "crossval_utils.py"): [
        (
            r"^from pl_utils import",
            "from neuro_green.green.research_code.pl_utils import",
        )
    ],
}


def main():
    if os.path.exists(TARGET):
        sys.exit(f"{TARGET} already exists; delete it first to reinstall.")
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    top = archive.namelist()[0].split("/")[0]
    staging = TARGET + "_download"
    archive.extractall(staging)
    shutil.move(os.path.join(staging, top), TARGET)
    shutil.rmtree(staging)
    for rel, rules in REWRITES.items():
        path = os.path.join(TARGET, rel)
        text = open(path, encoding="utf-8").read()
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text, flags=re.M)
        open(path, "w", encoding="utf-8", newline="").write(text)
    print(f"GREEN {COMMIT[:7]} installed in {TARGET}")


if __name__ == "__main__":
    main()
