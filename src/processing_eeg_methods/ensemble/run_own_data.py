"""Run the benchmark protocol on your own epoched EEG, with no code changes.

Put one file per subject in Datasets/<name>/ (or --data-dir):

  subject_<id>.npz      X       (trials, channels, samples), trials in RECORDING ORDER
                        y       (trials,) labels, numbers or strings
                        sfreq   sampling rate in Hz
                        session (trials,) optional, the session of each trial
                        run     (trials,) optional, the run/block of each trial
  subject_<id>-epo.fif  MNE Epochs; labels from the event names, session/run from
                        epochs.metadata columns "session"/"run" if present

Exactly as in the paper (benchmark_moabb.py / benchmark_extra.py):
  - train on the first session and test on the later ones; with one session, train on
    the first half of the runs (or trials) and test on the second half;
  - out-of-fold estimates with GroupKFold by run inside the training data;
  - data resampled to 128 Hz; voters, folds, combination rules and the baseline (the
    single voter picked on training out-of-fold accuracy) are the benchmark's own code.
Unlike MOABB, which filters the continuous recording before epoching, this filters the
epochs (1-45 Hz by default). If you already filtered the continuous data, pass --no-filter.

Usage:
  python -m processing_eeg_methods.ensemble.run_own_data <name> [--voters fast|v1|v2|a,b,c]
         [--subjects 1 2] [--fmin 1 --fmax 45 | --no-filter] [--data-dir Datasets]

Output: Results/own_data/<name>/<voters>/subject_<id>.npz (saved probabilities, same format
as the benchmark) and per_subject.csv; a summary is printed.
"""

import argparse
import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from processing_eeg_methods.ensemble.analyze_voters import (
    best_single,
    ensemble_accuracy,
    load_subject,
)
from processing_eeg_methods.ensemble.benchmark_extra import sync_gpu, training_groups
from processing_eeg_methods.ensemble.benchmark_moabb import SFREQ, split_sessions
from processing_eeg_methods.ensemble.features import bandpass
from processing_eeg_methods.ensemble.voter_sets import get_voters, voter_names
from processing_eeg_methods.ensemble.voting_ensemble import (
    _fit_one,
    _oof_one,
    _proba_full,
)
from processing_eeg_methods.share import ROOT_VOTING_SYSTEM_PATH

RULES = ["auto", "weighted", "stacking", "soft", "majority"]


def load_file(path):
    """Returns X, y, sfreq, meta (columns session, run) for one subject file."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        X, y, sfreq = z["X"], z["y"], float(z["sfreq"])
        session = z["session"] if "session" in z.files else np.zeros(len(y), int)
        run = z["run"] if "run" in z.files else np.zeros(len(y), int)
    else:
        import mne

        epochs = mne.read_epochs(path, preload=True, verbose="error")
        X, sfreq = epochs.get_data(), float(epochs.info["sfreq"])
        names = {code: name for name, code in epochs.event_id.items()}
        y = np.array([names[c] for c in epochs.events[:, 2]])
        md = epochs.metadata
        session = (
            md["session"].to_numpy()
            if md is not None and "session" in md
            else np.zeros(len(y), int)
        )
        run = (
            md["run"].to_numpy()
            if md is not None and "run" in md
            else np.zeros(len(y), int)
        )
    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(
            f"{path}: X must be (trials, channels, samples) with one label per trial"
        )
    meta = pd.DataFrame(
        {"session": np.asarray(session).astype(str), "run": np.asarray(run).astype(str)}
    )
    return np.asarray(X, dtype=np.float64), np.asarray(y), sfreq, meta


def preprocess(X, sfreq, fmin=1.0, fmax=45.0, filter_data=True):
    if filter_data:
        X = bandpass(X, fmin, min(fmax, 0.45 * sfreq), sfreq)
    if sfreq != SFREQ:
        import mne

        X = mne.filter.resample(X, up=SFREQ, down=sfreq, axis=-1, verbose="error")
    return X


def evaluate(X, y, meta, voters, out_path):
    """Trains every voter with the benchmark's splits and folds and saves the probabilities
    in the benchmark's format. Returns None if a class is missing from train or test."""
    train, test, _ = split_sessions(meta)
    if set(y[train]) != set(y) or set(y[test]) != set(y):  # PROTOCOL.md exclusion rule
        return None
    groups = training_groups(meta, train)
    encoder = LabelEncoder().fit(y[train])
    ytr, yte = encoder.transform(y[train]), encoder.transform(y[test])
    Xtr, Xte = X[train], X[test]
    n_classes = len(encoder.classes_)
    splits = list(
        GroupKFold(min(5, len(np.unique(groups[train])))).split(Xtr, ytr, groups[train])
    )
    saved = {
        "y_test": yte,
        "y_oof": ytr,
        "classes": np.asarray(encoder.classes_, dtype=str),
    }
    seconds = {}
    for name, voter in voters.items():
        t0 = time.perf_counter()
        saved[f"oof__{name}"] = _oof_one(voter, Xtr, ytr, splits, n_classes)
        model = _fit_one(voter, Xtr, ytr)
        sync_gpu()
        seconds[name] = time.perf_counter() - t0
        saved[f"test__{name}"] = _proba_full(model, Xte, n_classes)
        print(
            f"  {name:32s} {seconds[name]:7.1f}s  test acc {(saved[f'test__{name}'].argmax(1) == yte).mean():.3f}",
            flush=True,
        )
    np.savez_compressed(out_path, **saved)
    return seconds


def score(path, names):
    s = load_subject(path)
    row = {f"ensemble: {rule}": ensemble_accuracy(s, names, rule) for rule in RULES}
    row["best single (picked on training data)"], row["picked voter"] = best_single(
        s, names
    )
    row.update({f"voter: {v}": ensemble_accuracy(s, [v], "soft") for v in names})
    row["n_train"], row["n_test"] = len(s["y_oof"]), len(s["y_test"])
    return row


def subject_files(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "subject_*.npz"))
        + glob.glob(os.path.join(folder, "subject_*-epo.fif"))
    )
    ids = [
        re.match(r"subject_(.+?)(?:-epo\.fif|\.npz)$", os.path.basename(f)).group(1)
        for f in files
    ]
    return dict(zip(ids, files))


def run(
    name,
    voters="fast",
    subjects=None,
    data_dir=None,
    out_dir=None,
    fmin=1.0,
    fmax=45.0,
    filter_data=True,
    epochs=300,
):
    folder = os.path.join(
        data_dir or os.path.join(ROOT_VOTING_SYSTEM_PATH, "Datasets"), name
    )
    files = subject_files(folder)
    if not files:
        raise FileNotFoundError(
            f"no subject_<id>.npz or subject_<id>-epo.fif files in {folder}"
        )
    names = voter_names(voters)
    label = voters if isinstance(voters, str) and "," not in voters else "custom"
    out = os.path.join(
        out_dir or os.path.join(ROOT_VOTING_SYSTEM_PATH, "Results", "own_data"),
        name,
        label,
    )
    os.makedirs(out, exist_ok=True)
    rows = []
    for sid, path in files.items():
        if subjects and sid not in {str(s) for s in subjects}:
            continue
        target = os.path.join(out, f"subject_{sid}.npz")
        if not os.path.exists(target):  # resumable
            X, y, sfreq, meta = load_file(path)
            X = preprocess(X, sfreq, fmin, fmax, filter_data)
            print(
                f"{name} subject {sid}: {X.shape[0]} trials, {X.shape[1]} channels, "
                f"{len(set(y))} classes",
                flush=True,
            )
            if evaluate(X, y, meta, get_voters(names, SFREQ, epochs), target) is None:
                print(
                    f"{name} subject {sid}: EXCLUDED (a class is missing from the train or test split)"
                )
                continue
        rows.append({"subject": sid, **score(target, names)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_subject.csv"), index=False)
    summarize(df)
    return df


def summarize(df):
    if df.empty:
        return
    gain = df["ensemble: auto"] - df["best single (picked on training data)"]
    print(
        f"\n{len(df)} subjects | best single voter {100 * df['best single (picked on training data)'].mean():.1f}% "
        f"| ensemble (auto) {100 * df['ensemble: auto'].mean():.1f}% "
        f"| gain {100 * gain.mean():+.1f} ± {100 * gain.std(ddof=1 if len(df) > 1 else 0):.1f} pp, "
        f"improved {(gain > 0).sum()}/{len(df)}"
    )
    if len(df) >= 6 and (gain != 0).any():
        print(
            f"one-sided Wilcoxon (ensemble > best single): p = {wilcoxon(gain, alternative='greater').pvalue:.2g}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="folder name inside --data-dir")
    parser.add_argument(
        "--voters",
        default="fast",
        help="fast (default), v1, v2, or comma-separated names",
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument(
        "--data-dir", default=None, help="default: <repository>/Datasets"
    )
    parser.add_argument(
        "--out-dir", default=None, help="default: <repository>/Results/own_data"
    )
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=45.0)
    parser.add_argument(
        "--no-filter", action="store_true", help="data already band-pass filtered"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="training epochs of the network voters"
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    run(
        args.name,
        args.voters,
        args.subjects,
        args.data_dir,
        args.out_dir,
        args.fmin,
        args.fmax,
        not args.no_filter,
        args.epochs,
    )


if __name__ == "__main__":
    main()
