"""Does voting help when there IS signal? Cross-session benchmark on public MOABB datasets.

For every subject: train on the FIRST recording session, test on all LATER sessions
(different days), which is leakage-safe by construction and matches real use. Inside
the training session, the ensemble's out-of-fold weights use runs as groups.

Reported per subject: every single voter, every combination rule, and the voter you
would have picked WITHOUT an ensemble (best out-of-fold voter) - the fair baseline.

Usage (needs moabb; deep voters need torch/braindecode, GREEN needs geotorch+lightning):
    python -m processing_eeg_methods.ensemble.benchmark_moabb BNCI2014_001 BNCI2014_004 ...
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from processing_eeg_methods.ensemble.voters import deep_voters, default_voters
from processing_eeg_methods.ensemble.voting_ensemble import VotingEnsemble
from processing_eeg_methods.share import ROOT_VOTING_SYSTEM_PATH

OUT_DIR = os.path.join(ROOT_VOTING_SYSTEM_PATH, "Results", "ensemble_benchmark")
SFREQ = 128.0


def get_dataset(name):
    from moabb import datasets as D
    from moabb import paradigms as P

    mi = dict(fmin=1, fmax=45, resample=SFREQ)
    table = {
        "BNCI2014_001": (D.BNCI2014_001(), P.MotorImagery(n_classes=4, **mi)),        # 4-class MI, 2 sessions
        "BNCI2014_004": (D.BNCI2014_004(), P.LeftRightImagery(**mi)),                 # 2-class MI, 5 sessions
        "Zhou2016": (D.Zhou2016(), P.MotorImagery(n_classes=3, **mi)),                # 3-class MI, 3 sessions
        "BNCI2015_001": (D.BNCI2015_001(), P.MotorImagery(n_classes=2, **mi)),        # 2-class MI, 2-3 sessions
        "Lee2019_MI": (D.Lee2019_MI(), P.LeftRightImagery(**mi)),                     # 2-class MI, 2 sessions, 54 subj
        "Cho2017": (D.Cho2017(), P.LeftRightImagery(**mi)),                           # 1 session -> run split
        "BNCI2014_009": (D.BNCI2014_009(), P.P300(fmin=1, fmax=24, resample=SFREQ)),  # P300, 3 sessions
        # ---- candidates for the confirmatory set (see PROTOCOL.md) ----
        "BNCI2015_004": (D.BNCI2015_004(), P.MotorImagery(n_classes=5, **mi)),        # 5-class mental tasks, 2 sessions
        "Shin2017A": (D.Shin2017A(accept=True), P.LeftRightImagery(**mi)),           # 2-class MI, 3 sessions
        "Schirrmeister2017": (D.Schirrmeister2017(), P.MotorImagery(n_classes=4, **mi)),  # HGD, train/test runs
        "PhysionetMI": (D.PhysionetMI(), P.LeftRightImagery(**mi)),                  # 109 subj, several runs
        "Weibo2014": (D.Weibo2014(), P.LeftRightImagery(**mi)),                      # 10 subj
        "EPFLP300": (D.EPFLP300(), P.P300(fmin=1, fmax=24, resample=SFREQ)),          # P300, 4 sessions
        "BNCI2015_003": (D.BNCI2015_003(), P.P300(fmin=1, fmax=24, resample=SFREQ)),  # P300
        "BI2013a": (D.BI2013a(), P.P300(fmin=1, fmax=24, resample=SFREQ)),            # P300, several sessions
        "MAMEM3": (D.MAMEM3(), P.SSVEP(n_classes=5, fmin=1, fmax=45, resample=SFREQ)),
        "Wang2016": (D.Wang2016(), P.SSVEP(n_classes=40, fmin=1, fmax=45, resample=SFREQ)),
        "Kalunga2016":(D.Kalunga2016(), P.SSVEP(n_classes=4, fmin=1, fmax=45, resample=SFREQ)),
        "Nakanishi2015": (D.Nakanishi2015(), P.SSVEP(n_classes=12, fmin=1, fmax=45, resample=SFREQ)),
    }
    return table[name]


def split_sessions(meta):
    """Train = first session; test = later sessions. If there is a single session, train on
    the first half of its runs and test on the second half (still chronological)."""
    sessions = sorted(meta["session"].unique())
    if len(sessions) > 1:
        train = (meta["session"] == sessions[0]).to_numpy()
        return train, ~train, meta["run"].astype(str).to_numpy()
    runs = sorted(meta["run"].unique())
    if len(runs) > 1:
        train = meta["run"].isin(runs[: len(runs) // 2]).to_numpy()
        return train, ~train, meta["run"].astype(str).to_numpy()
    n = len(meta)
    order = np.arange(n)
    train = order < n // 2
    return train, ~train, (order * 10 // n).astype(str)


def evaluate_subject(X, y, meta, use_deep=True, deep_epochs=300):
    train, test, groups = split_sessions(meta)
    if len(np.unique(groups[train])) < 3:  # not enough runs to group by: use contiguous blocks
        idx = np.flatnonzero(train)
        groups = groups.copy()
        groups[idx] = (np.arange(len(idx)) * 6 // len(idx)).astype(str)
    voters = default_voters(SFREQ)
    if use_deep:
        voters.update(deep_voters(SFREQ, epochs=deep_epochs))
    rows = {}
    ens = VotingEnsemble(voters, combination="weighted", n_jobs=1).fit(X[train], y[train], groups=groups[train])
    probas = ens.voter_proba(X[test])
    y_test = ens.encoder_.transform(y[test])
    saved = {"y_test": y_test, "y_oof": ens._oof_y, "classes": np.asarray(ens.classes_, dtype=str)}
    saved.update({f"test__{n}": p for n, p in probas.items()})
    saved.update({f"oof__{n}": p for n, p in ens.oof_proba_.items()})
    for n, p in probas.items():
        rows[f"voter: {n}"] = balanced_accuracy_score(y_test, p.argmax(1))
    enough_for_stacking = np.bincount(ens._oof_y).min() >= ens.min_trials_per_class_for_stacking
    if enough_for_stacking:
        from sklearn.linear_model import LogisticRegression

        ens.meta_ = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(
            ens._meta_features(ens.oof_proba_), ens._oof_y)
    for combination in ["weighted", "soft", "majority", "stacking"]:
        used = combination if (combination != "stacking" or enough_for_stacking) else "weighted"
        rows[f"ensemble: {combination}"] = balanced_accuracy_score(y_test, ens._combine(probas, used).argmax(1))
    best = max(ens.oof_accuracy_, key=ens.oof_accuracy_.get)
    rows["baseline: best single voter (picked on training session)"] = rows[f"voter: {best}"]
    report = ens.diversity_report()
    rows["mean error correlation"] = float(report["mean_error_correlation_with_others"].mean())
    return rows, best, saved


def run_dataset(name):
    dataset, paradigm = get_dataset(name)
    out_csv = os.path.join(OUT_DIR, f"{name}.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["subject"])
    for subject in dataset.subject_list:
        if subject in set(done["subject"]):
            continue
        t = time.time()
        X, y, meta = paradigm.get_data(dataset, [subject])
        train, test, _ = split_sessions(meta)
        if set(y[train]) != set(y) or set(y[test]) != set(y):  # PROTOCOL.md exclusion rule
            print(f"{name} subject {subject}: EXCLUDED (a class is missing from the train or test split)", flush=True)
            with open(os.path.join(OUT_DIR, "excluded_subjects.txt"), "a") as f:
                f.write(f"{name}\t{subject}\tclass missing from train or test split\n")
            continue
        rows, best, saved = evaluate_subject(X, y, meta)
        os.makedirs(os.path.join(OUT_DIR, "probabilities", name), exist_ok=True)
        np.savez_compressed(os.path.join(OUT_DIR, "probabilities", name, f"subject_{subject}.npz"), **saved)
        rows.update({"dataset": name, "subject": subject, "picked_voter": best,
                     "n_train": int(split_sessions(meta)[0].sum()), "n_test": int(split_sessions(meta)[1].sum()),
                     "minutes": round((time.time() - t) / 60, 1)})
        done = pd.concat([done, pd.DataFrame([rows])], ignore_index=True)
        done.to_csv(out_csv, index=False)  # incremental, resumable
        print(f"{name} subject {subject}: best-single {rows['baseline: best single voter (picked on training session)']:.3f} "
              f"| weighted {rows['ensemble: weighted']:.3f} | stacking {rows['ensemble: stacking']:.3f} "
              f"({rows['minutes']} min)", flush=True)
    summary = done.drop(columns=["subject", "picked_voter", "dataset", "n_train", "n_test", "minutes"],
                        errors="ignore").mean(numeric_only=True).sort_values(ascending=False)
    print(f"\n=== {name}: mean over {len(done)} subjects ===\n{summary.round(3).to_string()}\n", flush=True)
    json.dump(summary.to_dict(), open(os.path.join(OUT_DIR, f"{name}_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in sys.argv[1:]:
        try:
            run_dataset(name)
        except Exception as error:  # one broken download/reader must not stop the others
            print(f"!!! {name} failed: {type(error).__name__}: {error}", flush=True)
