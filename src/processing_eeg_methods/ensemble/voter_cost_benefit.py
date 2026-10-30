"""Exploratory (Amendment A1, descriptive): what each v2 voter adds, and what it costs.

For every subject with all v2 voters, removes one voter at a time from v2 and records the change
in auto-ensemble accuracy; joins each voter's median calibration time and latency.

Writes Results/ensemble_benchmark/analysis/v2_leave_one_out.csv (one row per subject and voter).

Usage: python -m processing_eeg_methods.ensemble.voter_cost_benefit
"""

import json
import os

import pandas as pd

from processing_eeg_methods.ensemble.analyze_voters import ensemble_accuracy
from processing_eeg_methods.ensemble.select_voters import CONFIRMATORY, DEVELOPMENT, OUT_DIR, V2_FILE, load_merged


def main():
    v2 = json.load(open(V2_FILE))["voters"]
    rows = []
    for d in DEVELOPMENT + CONFIRMATORY:
        for s in load_merged(d):
            if not set(v2) <= set(s["voters"]):
                print(f"skip {d} {s['subject']}: v2 voters missing")
                continue
            full = ensemble_accuracy(s, v2, "auto")
            for v in v2:
                rows.append({"dataset": d, "subject": s["subject"],
                             "set": "Confirmatory" if d in CONFIRMATORY else "Development",
                             "removed": v, "full": full,
                             "delta_when_removed": ensemble_accuracy(s, [u for u in v2 if u != v], "auto") - full})
        print(d, "done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "analysis", "v2_leave_one_out.csv"), index=False)
    print(df.groupby(["removed", "set"]).delta_when_removed.agg(["mean", "std"]).unstack().mul(100).round(2))


if __name__ == "__main__":
    main()
