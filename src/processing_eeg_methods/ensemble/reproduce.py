"""Reproduce the paper's benchmark, in the order it was run.

  python -m processing_eeg_methods.ensemble.reproduce check
      code fingerprints against PROTOCOL.md, environment against the lock file, GPU, GREEN
  python -m processing_eeg_methods.ensemble.reproduce benchmark [--datasets ...]
      1. benchmark_moabb: the 12 pre-registered voters (v1), every subject
      2. benchmark_extra: the 8 extra voters of Amendment A1, same splits and folds
  python -m processing_eeg_methods.ensemble.reproduce timing [--datasets ...]
      calibration time and single-trial latency of all 20 voters, subject 1 of each dataset,
      with nothing else running on the machine
  python -m processing_eeg_methods.ensemble.reproduce analyze
      H1-H3 and ablations (analyze_voters), v2 selection on development data
      (select_voters select), H4-H5 (select_voters evaluate), leave-one-voter-out
      (voter_cost_benefit), and whether one voter chosen in advance could replace the ensemble
      (explore_selection, exploratory)
  python -m processing_eeg_methods.ensemble.reproduce all
      benchmark, then timing, then analyze

Every step is resumable: finished subjects are skipped. Results go to Results/ensemble_benchmark/.
MOABB downloads the datasets on first use to MNE_DATA (default ~/mne_data, 24 GB in total).
Compute: about 75 GPU-hours for step 1 and 45 for step 2 on one 8 GB laptop GPU, most of it
Schirrmeister2017. Network voters are not bit-for-bit deterministic on GPU, so expect the
same conclusions with small differences in individual accuracies.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from importlib import metadata

from processing_eeg_methods.share import ROOT_VOTING_SYSTEM_PATH

DEVELOPMENT = [
    "BNCI2014_001",
    "BNCI2014_004",
    "BNCI2015_001",
    "BNCI2014_009",
    "Kalunga2016",
]
CONFIRMATORY = [
    "BNCI2015_004",
    "Schirrmeister2017",
    "Weibo2014",
    "BNCI2015_003",
    "MAMEM3",
]
TIMING_ORDER = [
    "BNCI2014_001",
    "BNCI2014_004",
    "BNCI2014_009",
    "BNCI2015_001",
    "Kalunga2016",
    "BNCI2015_004",
    "Weibo2014",
    "BNCI2015_003",
    "MAMEM3",
    "Schirrmeister2017",
]
PROTOCOL = os.path.join(os.path.dirname(__file__), "PROTOCOL.md")
LOCK = os.path.join(ROOT_VOTING_SYSTEM_PATH, "requirements-ensemble.lock.txt")
KEY_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "pyriemann",
    "mne",
    "moabb",
    "torch",
    "braindecode",
    "lightning",
    "geotorch",
    "sktime",
    "antropy",
]


def file_hash(path):
    """SHA-256 (first 16 hex) with line endings normalised, so it is the same on every OS."""
    data = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def recorded_fingerprints():
    """The line-ending-independent hashes from PROTOCOL.md ('Fingerprints at public release')."""
    text = open(PROTOCOL, encoding="utf-8").read()
    section = text.split("### Fingerprints at public release", 1)[1].split("\n#", 1)[0]
    return dict(
        re.findall(
            r"\|\s*`([^`]+)`\s*\|\s*`[0-9a-f]{16}`\s*\|\s*`([0-9a-f]{16})`\s*\|",
            section,
        )
    )


def check_fingerprints():
    ok = True
    for rel, expected in recorded_fingerprints().items():
        path = os.path.join(ROOT_VOTING_SYSTEM_PATH, rel)
        got = file_hash(path) if os.path.exists(path) else "missing"
        ok &= got == expected
        print(f"  {'ok  ' if got == expected else 'DIFF'} {rel} {got}")
    return ok


def check_environment():
    if not os.path.exists(LOCK):
        print(f"  {LOCK} not found: run from a clone of the repository")
        return False
    pinned = dict(
        re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", open(LOCK).read(), re.M)
    )
    pinned = {k.lower().replace("_", "-"): v for k, v in pinned.items()}
    same = True
    for name in KEY_PACKAGES:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        want = pinned.get(name, "?")
        same &= have == want
        print(
            f"  {'ok  ' if have == want else 'DIFF'} {name:14s} installed {have:16s} paper {want}"
        )
    print(f"  python {sys.version.split()[0]} (paper: 3.10)")
    return same


def check_green():
    sys.path.insert(0, ROOT_VOTING_SYSTEM_PATH)
    try:
        from neuro_green.green.research_code.pl_utils import get_green  # noqa: F401

        return True
    except Exception as error:  # noqa: BLE001
        print(
            f"  GREEN unavailable ({type(error).__name__}: {error}). Run: python tools/install_green.py"
        )
        return False


def check():
    print("Code fingerprints (PROTOCOL.md):")
    code = check_fingerprints()
    print("Environment (requirements-ensemble.lock.txt):")
    env = check_environment()
    green = check_green()
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    print(
        f"\ncode identical to the paper: {code}\nenvironment identical to the paper: {env}\n"
        f"GREEN installed: {green}\nCUDA GPU: {gpu}"
    )
    if not env:
        print(
            "A different environment reproduces the protocol, not every decimal. For the paper's "
            "exact environment see README, 'Replicate the paper'."
        )
    return code


def _quiet():
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")


def benchmark(datasets):
    from processing_eeg_methods.ensemble import benchmark_extra, benchmark_moabb
    from processing_eeg_methods.ensemble.extra_voters import extra_voters

    _quiet()
    os.makedirs(benchmark_moabb.OUT_DIR, exist_ok=True)
    for name in datasets:
        benchmark_moabb.run_dataset(name)
    benchmark_extra.run(
        datasets, list(extra_voters(benchmark_moabb.SFREQ)), None, "probabilities_extra"
    )


def timing(datasets):
    from processing_eeg_methods.ensemble import benchmark_extra
    from processing_eeg_methods.ensemble.benchmark_moabb import SFREQ
    from processing_eeg_methods.ensemble.extra_voters import extra_voters
    from processing_eeg_methods.ensemble.voters import deep_voters, default_voters

    _quiet()
    ordered = [d for d in TIMING_ORDER if d in datasets] + [
        d for d in datasets if d not in TIMING_ORDER
    ]
    benchmark_extra.run(ordered, list(extra_voters(SFREQ)), [1], "timing")
    benchmark_extra.run(
        ordered, list({**default_voters(SFREQ), **deep_voters(SFREQ)}), [1], "timing"
    )


def analyze():
    from processing_eeg_methods.ensemble import (
        analyze_voters,
        explore_selection,
        select_voters,
        voter_cost_benefit,
    )
    from processing_eeg_methods.ensemble.voter_sets import V2

    present = sorted(os.listdir(os.path.join(select_voters.OUT_DIR, "probabilities")))
    analyze_voters.main(present)
    select_voters.select()
    chosen = json.load(open(select_voters.V2_FILE))["voters"]
    print(
        "v2 selection "
        + (
            "matches the paper."
            if sorted(chosen) == sorted(V2)
            else f"DIFFERS from the paper's {sorted(V2)}; the evaluation uses this run's selection."
        )
    )
    select_voters.evaluate()
    voter_cost_benefit.main()
    explore_selection.main()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "step", choices=["check", "benchmark", "timing", "analyze", "all"]
    )
    parser.add_argument("--datasets", nargs="+", default=DEVELOPMENT + CONFIRMATORY)
    args = parser.parse_args(argv)
    if args.step == "check":
        sys.exit(0 if check() else 1)
    if args.step in ("benchmark", "all"):
        benchmark(args.datasets)
    if args.step in ("timing", "all"):
        timing(args.datasets)
    if args.step in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
