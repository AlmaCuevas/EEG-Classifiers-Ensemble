"""Amendment A1 (PROTOCOL.md): choose the v2 voter set on development data only, then test it.

  python -m processing_eeg_methods.ensemble.select_voters select
      backward elimination over the 20-voter pool on the development datasets;
      writes Results/ensemble_benchmark/v2_voters.json
  python -m processing_eeg_methods.ensemble.select_voters evaluate
      H4 / H5 on the confirmatory datasets with the frozen v2 list;
      writes Results/ensemble_benchmark/analysis/v2_per_subject.csv
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from processing_eeg_methods.ensemble.analyze_voters import best_single, ensemble_accuracy, load_subject
from processing_eeg_methods.ensemble.benchmark_moabb import OUT_DIR
from processing_eeg_methods.ensemble.extra_voters import DEEP_EXTRA

DEVELOPMENT = ["BNCI2014_001", "BNCI2014_004", "BNCI2015_001", "BNCI2014_009", "Kalunga2016"]
CONFIRMATORY = ["BNCI2015_004", "Schirrmeister2017", "Weibo2014", "BNCI2015_003", "MAMEM3"]
EXTRA_DIR = os.path.join(OUT_DIR, "probabilities_extra")
V2_FILE = os.path.join(OUT_DIR, "v2_voters.json")
DEEP = {"eegnet", "shallownet", "green"} | DEEP_EXTRA
MIN_IMPROVEMENT = 0.001  # 0.1 percentage points


def load_merged(dataset):
    """Frozen-run probabilities plus every extra voter saved for the same subject."""
    subjects = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "probabilities", dataset, "subject_*.npz"))):
        s = load_subject(path)
        sid = os.path.basename(path)[len("subject_"):-len(".npz")]
        for extra in glob.glob(os.path.join(EXTRA_DIR, dataset, f"subject_{sid}__*.npz")):
            voter = os.path.basename(extra).split("__", 1)[1][:-len(".npz")]
            z = np.load(extra)
            assert np.array_equal(z["y_test"], s["y_test"]) and np.array_equal(z["y_oof"], s["y_oof"]), extra
            s["test"][voter], s["oof"][voter] = z["test"], z["oof"]
            s["voters"].append(voter)
        s["dataset"], s["subject"] = dataset, sid
        subjects.append(s)
    return subjects


def criterion(subjects, voters):
    return float(np.mean([ensemble_accuracy(s, voters, "auto") for s in subjects]))


def select():
    subjects = [s for d in DEVELOPMENT for s in load_merged(d)]
    pool = sorted(set.intersection(*(set(s["voters"]) for s in subjects)))
    print(f"{len(subjects)} development subjects, pool of {len(pool)} voters: {pool}")
    current, score = list(pool), criterion(subjects, pool)
    history = [{"step": 0, "removed": None, "n_voters": len(current), "criterion": score}]
    print(f"all {len(current)} voters: {100 * score:.2f}%")
    while len(current) > 2:
        trials = {v: criterion(subjects, [u for u in current if u != v]) for v in current}
        best = max(trials, key=trials.get)
        if trials[best] - score < MIN_IMPROVEMENT:
            break
        current.remove(best)
        score = trials[best]
        history.append({"step": len(history), "removed": best, "n_voters": len(current), "criterion": score})
        print(f"remove {best:28s} -> {100 * score:.2f}%")
    json.dump({"voters": current, "pool": pool, "history": history}, open(V2_FILE, "w"), indent=1)
    print(f"v2 ({len(current)} voters): {current}")


def evaluate():
    v2 = json.load(open(V2_FILE))["voters"]
    v1 = ["coherence", "complexity_boosting", "covariance_tangent_space", "covariance_tangent_space_4_30Hz",
          "csp_lda_8_30Hz", "eegnet", "erp_time_course", "green", "minirocket", "relative_spectrum",
          "riemann_mdm_8_30Hz", "shallownet"]
    fast = [v for v in v2 if v not in DEEP]
    rows = []
    for d in CONFIRMATORY + DEVELOPMENT:
        for s in load_merged(d):
            if not set(v2) <= set(s["voters"]):
                print(f"skip {d} {s['subject']}: extra voters missing")
                continue
            base_v2, picked = best_single(s, v2)
            rows.append({"dataset": d, "subject": s["subject"],
                         "set": "Confirmatory" if d in CONFIRMATORY else "Development",
                         "v1_ensemble": ensemble_accuracy(s, v1, "auto"), "v1_best_single": best_single(s, v1)[0],
                         "v2_ensemble": ensemble_accuracy(s, v2, "auto"), "v2_best_single": base_v2,
                         "v2_picked": picked, "fast_ensemble": ensemble_accuracy(s, fast, "auto"),
                         "fast_best_single": best_single(s, fast)[0]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT_DIR, "analysis"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "analysis", "v2_per_subject.csv"), index=False)
    for label, g in df.groupby("set"):
        gain4 = g.v2_ensemble - g.v2_best_single
        diff5 = g.v2_ensemble - g.v1_ensemble
        print(f"\n=== {label} (n={len(g)}) ===")
        print(f"v1 ensemble {100 * g.v1_ensemble.mean():.1f}%  v2 ensemble {100 * g.v2_ensemble.mean():.1f}%  "
              f"fast ensemble {100 * g.fast_ensemble.mean():.1f}%")
        print(f"H4 v2 vs its best single: {100 * gain4.mean():+.1f} ± {100 * gain4.std():.1f} pp, "
              f"improved {(gain4 > 0).sum()}/{len(g)}, p={wilcoxon(gain4, alternative='greater').pvalue:.1e}")
        print(f"H5 v2 vs v1 ensemble:     {100 * diff5.mean():+.1f} ± {100 * diff5.std():.1f} pp, "
              f"p={wilcoxon(diff5).pvalue:.1e}" if (diff5 != 0).any() else "H5: identical")


if __name__ == "__main__":
    {"select": select, "evaluate": evaluate}[sys.argv[1]]()
