"""Train extra (or timing-only) voters with exactly the splits and folds of benchmark_moabb,
so their probabilities can be merged with the frozen run without retraining anything.

For every voter it also measures:
  oof_seconds   time of the 5-fold out-of-fold estimate (calibration cost)
  fit_seconds   time to fit on the whole training session (calibration cost)
  latency_ms    median time to classify ONE new trial (online cost)

Output: Results/ensemble_benchmark/<out>/<dataset>/subject_<id>__<voter>.npz
        Results/ensemble_benchmark/<out>/timing.csv

Usage:
  python -m processing_eeg_methods.ensemble.benchmark_extra --datasets BNCI2014_001 ... \
         [--voters xdawn_tangent atcnet ...] [--subjects 1 2] [--out probabilities_extra]
  --voters frozen  times the 12 pre-registered voters instead (use with --out timing)
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from processing_eeg_methods.ensemble.benchmark_moabb import OUT_DIR, SFREQ, get_dataset, split_sessions
from processing_eeg_methods.ensemble.extra_voters import extra_voters
from processing_eeg_methods.ensemble.voters import deep_voters, default_voters
from processing_eeg_methods.ensemble.voting_ensemble import _fit_one, _oof_one, _proba_full


def training_groups(meta, train):
    """Identical to benchmark_moabb.evaluate_subject."""
    _, _, groups = split_sessions(meta)
    if len(np.unique(groups[train])) < 3:
        idx = np.flatnonzero(train)
        groups = groups.copy()
        groups[idx] = (np.arange(len(idx)) * 6 // len(idx)).astype(str)
    return groups


def sync_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def run(datasets, voter_names, subjects, out):
    root = os.path.join(OUT_DIR, out)
    timing_csv = os.path.join(root, "timing.csv")
    os.makedirs(root, exist_ok=True)
    for name in datasets:
        dataset, paradigm = get_dataset(name)
        for subject in subjects or dataset.subject_list:
            if subject not in dataset.subject_list:
                continue
            target = os.path.join(root, name)
            todo = [v for v in voter_names
                    if not os.path.exists(os.path.join(target, f"subject_{subject}__{v}.npz"))]
            if not todo:
                continue
            X, y, meta = paradigm.get_data(dataset, [subject])
            train, test, _ = split_sessions(meta)
            if set(y[train]) != set(y) or set(y[test]) != set(y):
                continue  # excluded by the pre-registered rule
            groups = training_groups(meta, train)
            encoder = LabelEncoder().fit(y[train])
            ytr, yte = encoder.transform(y[train]), encoder.transform(y[test])
            Xtr, Xte = X[train], X[test]
            n_classes = len(encoder.classes_)
            splits = list(GroupKFold(min(5, len(np.unique(groups[train])))).split(Xtr, ytr, groups[train]))
            pool = {**default_voters(SFREQ), **deep_voters(SFREQ, epochs=300), **extra_voters(SFREQ)}
            os.makedirs(target, exist_ok=True)
            for v in todo:
                t0 = time.perf_counter()
                oof = _oof_one(pool[v], Xtr, ytr, splits, n_classes)
                sync_gpu()
                t1 = time.perf_counter()
                model = _fit_one(pool[v], Xtr, ytr)
                sync_gpu()
                t2 = time.perf_counter()
                proba = _proba_full(model, Xte, n_classes)
                latencies = []
                for i in range(min(30, len(Xte))):  # one trial at a time, as online
                    s = time.perf_counter()
                    _proba_full(model, Xte[i : i + 1], n_classes)
                    sync_gpu()
                    latencies.append(time.perf_counter() - s)
                row = {"dataset": name, "subject": subject, "voter": v, "oof_seconds": t1 - t0,
                       "fit_seconds": t2 - t1, "latency_ms": 1000 * float(np.median(latencies[1:] or latencies)),
                       "n_train": len(ytr), "n_channels": X.shape[1], "n_times": X.shape[2],
                       "epoch_seconds": X.shape[2] / SFREQ}
                np.savez_compressed(os.path.join(target, f"subject_{subject}__{v}.npz"), test=proba, oof=oof,
                                    y_test=yte, y_oof=ytr, **{k: row[k] for k in ["oof_seconds", "fit_seconds",
                                                                                   "latency_ms"]})
                pd.DataFrame([row]).to_csv(timing_csv, mode="a", header=not os.path.exists(timing_csv), index=False)
                print(f"{name} s{subject} {v:24s} oof {row['oof_seconds']:6.1f}s  fit {row['fit_seconds']:6.1f}s  "
                      f"latency {row['latency_ms']:7.1f}ms  test acc "
                      f"{(proba.argmax(1) == yte).mean():.3f}", flush=True)


if __name__ == "__main__":
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--voters", nargs="+", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--out", default="probabilities_extra")
    args = parser.parse_args()
    names = args.voters or list(extra_voters(SFREQ))
    if names == ["frozen"]:
        names = list({**default_voters(SFREQ), **deep_voters(SFREQ)})
    run(args.datasets, names, args.subjects, args.out)
