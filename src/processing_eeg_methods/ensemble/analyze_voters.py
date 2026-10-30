"""Ablations for the voting-ensemble paper, computed from saved voter probabilities
(Results/ensemble_benchmark/probabilities/<dataset>/subject_*.npz, written by
benchmark_moabb.py). Nothing is retrained, so every analysis uses exactly the same voters.

All choices (weights, which single voter is "best", stacking combiner) use only the
out-of-fold predictions on the TRAINING session; the test session is only scored.

Outputs (Results/ensemble_benchmark/analysis/):
  per_subject.csv     ensemble vs best-single for every subject and combination rule
  n_voters.csv        gain as a function of the number of voters (random subsets)
  leave_one_out.csv   change in ensemble accuracy when each voter is removed
  diversity.csv       per-subject error correlation vs gain
  families.csv        classical-only vs deep-only vs all voters

Usage: python -m processing_eeg_methods.ensemble.analyze_voters [dataset ...]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from processing_eeg_methods.ensemble.benchmark_moabb import OUT_DIR

DEEP = {"eegnet", "shallownet", "green"}
MIN_TRIALS_PER_CLASS_FOR_STACKING = 40


def load_subject(path):
    z = np.load(path)
    voters = sorted(k[len("test__"):] for k in z.files if k.startswith("test__"))
    return {
        "voters": voters,
        "y_test": z["y_test"], "y_oof": z["y_oof"],
        "test": {v: z[f"test__{v}"] for v in voters},
        "oof": {v: z[f"oof__{v}"] for v in voters},
    }


def acc(y, p):
    return balanced_accuracy_score(y, p.argmax(1))


def ensemble_accuracy(s, voters, rule):
    """Re-implements VotingEnsemble's combination rules on a subset of voters."""
    n_classes = s["oof"][voters[0]].shape[1]
    chance = 1 / n_classes
    if rule == "auto":
        enough = np.bincount(s["y_oof"]).min() >= MIN_TRIALS_PER_CLASS_FOR_STACKING
        rule = "stacking" if enough and len(voters) > 1 else "weighted"
    test = np.stack([s["test"][v] for v in voters])
    if rule == "soft":
        return acc(s["y_test"], test.mean(0))
    if rule == "weighted":
        w = np.array([max(acc(s["y_oof"], s["oof"][v]) - chance, 0) for v in voters])
        w = w / w.sum() if w.sum() > 0 else np.full(len(voters), 1 / len(voters))
        return acc(s["y_test"], np.tensordot(w, test, 1))
    if rule == "majority":
        votes = test.argmax(2)
        counts = np.stack([(votes == c).sum(0) for c in range(n_classes)], 1) + 1e-3 * test.mean(0)
        return acc(s["y_test"], counts)
    if rule == "stacking":
        feats = lambda d: np.log(np.concatenate([d[v] for v in voters], axis=1) + 1e-9)  # noqa: E731
        meta = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(feats(s["oof"]), s["y_oof"])
        return balanced_accuracy_score(s["y_test"], meta.predict(feats(s["test"])))
    raise ValueError(rule)


def best_single(s, voters):
    best = max(voters, key=lambda v: acc(s["y_oof"], s["oof"][v]))
    return acc(s["y_test"], s["test"][best]), best


def error_correlation(s, voters):
    correct = np.array([s["oof"][v].argmax(1) == s["y_oof"] for v in voters], dtype=float)
    c = np.corrcoef(correct)
    return float(np.nanmean(c[np.triu_indices(len(voters), 1)]))


def main(datasets):
    out = os.path.join(OUT_DIR, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(42)
    per_subject, n_voters, loo, diversity, families = [], [], [], [], []
    for ds in datasets:
        for path in sorted(glob.glob(os.path.join(OUT_DIR, "probabilities", ds, "subject_*.npz"))):
            s = load_subject(path)
            subject = os.path.basename(path)[8:-4]
            voters = s["voters"]
            base, picked = best_single(s, voters)
            row = {"dataset": ds, "subject": subject, "best_single": base, "picked": picked,
                   "oracle_best_voter": max(acc(s["y_test"], s["test"][v]) for v in voters)}
            for rule in ["auto", "weighted", "soft", "majority", "stacking"]:
                row[rule] = ensemble_accuracy(s, voters, rule)
            per_subject.append(row)
            full = row["auto"]
            for k in range(1, len(voters) + 1):  # random subsets of k voters
                for _ in range(20 if k < len(voters) else 1):
                    sub = list(rng.choice(voters, k, replace=False))
                    n_voters.append({"dataset": ds, "subject": subject, "k": k,
                                     "gain": ensemble_accuracy(s, sub, "auto") - best_single(s, sub)[0],
                                     "accuracy": ensemble_accuracy(s, sub, "auto")})
            for v in voters:
                rest = [u for u in voters if u != v]
                loo.append({"dataset": ds, "subject": subject, "removed": v,
                            "delta_when_removed": ensemble_accuracy(s, rest, "auto") - full})
            diversity.append({"dataset": ds, "subject": subject, "error_correlation": error_correlation(s, voters),
                              "gain": full - base})
            for fam, members in [("classical", [v for v in voters if v not in DEEP]),
                                 ("deep", [v for v in voters if v in DEEP]), ("all", voters)]:
                if members:
                    families.append({"dataset": ds, "subject": subject, "family": fam,
                                     "ensemble": ensemble_accuracy(s, members, "auto"),
                                     "best_single": best_single(s, members)[0]})
        print(f"analysed {ds}", flush=True)

    ps = pd.DataFrame(per_subject); ps.to_csv(os.path.join(out, "per_subject.csv"), index=False)
    nv = pd.DataFrame(n_voters); nv.to_csv(os.path.join(out, "n_voters.csv"), index=False)
    lo = pd.DataFrame(loo); lo.to_csv(os.path.join(out, "leave_one_out.csv"), index=False)
    dv = pd.DataFrame(diversity); dv.to_csv(os.path.join(out, "diversity.csv"), index=False)
    fm = pd.DataFrame(families); fm.to_csv(os.path.join(out, "families.csv"), index=False)

    print("\n=== Ensemble ('auto') vs best single voter picked on training data ===")
    for ds, g in ps.groupby("dataset"):
        gain = g["auto"] - g["best_single"]
        print(f"{ds:18s} n={len(g):3d}  best-single {g.best_single.mean():.3f}  ensemble {g['auto'].mean():.3f}  "
              f"gain {gain.mean():+.3f}  improved {(gain > 0).sum()}/{len(g)}")
    gain = ps["auto"] - ps["best_single"]
    print(f"ALL                n={len(ps):3d}  mean gain {gain.mean():+.3f}  improved {(gain > 0).sum()}/{len(ps)}  "
          f"Wilcoxon p={wilcoxon(gain, alternative='greater').pvalue:.1e}")
    print("\n=== Gain vs number of voters (mean over subjects) ===")
    print(nv.groupby("k")[["gain", "accuracy"]].mean().round(3).to_string())
    print("\n=== Leave-one-voter-out: mean change in ensemble accuracy (negative = voter helps) ===")
    print(lo.groupby("removed")["delta_when_removed"].mean().sort_values().round(4).to_string())
    r = spearmanr(dv["error_correlation"], dv["gain"])
    print(f"\n=== Diversity: Spearman(error correlation, gain) = {r.statistic:.3f} (p={r.pvalue:.1e}) ===")
    print("\n=== Families ===")
    print(fm.groupby("family")[["ensemble", "best_single"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(os.path.join(OUT_DIR, "probabilities"))))
