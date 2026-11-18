"""Amendment A2 (PROTOCOL.md): tune the networks on development data, then test once.

  python -m processing_eeg_methods.ensemble.tune_networks tune
      every network x every configuration on the 5 development datasets (same trials,
      chronological split and grouped 5-fold OOF as the benchmark)
  python -m processing_eeg_methods.ensemble.tune_networks select
      per paradigm and network, the configuration with the highest development OOF balanced
      accuracy (mean over subjects, then over datasets) -> selected.json. Test data is not read.
  python -m processing_eeg_methods.ensemble.tune_networks confirm
      the selected configuration of every network, once, on the 5 confirmatory datasets
  python -m processing_eeg_methods.ensemble.tune_networks analyze
      the planned comparisons -> RESULTS.md and results_*.csv

The "paper" configuration is not retrained: its probabilities are read from the benchmark run
(probabilities/ and probabilities_extra/). Needs the benchmark results first. Outputs go to
Results/ensemble_benchmark/tuning/. Compute: about 13 days on one 8 GB laptop GPU, most of it
the 1000-epoch configurations on Schirrmeister2017.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from processing_eeg_methods.share import ROOT_VOTING_SYSTEM_PATH

BENCH = os.path.join(ROOT_VOTING_SYSTEM_PATH, "Results", "ensemble_benchmark")
OUT = os.path.join(BENCH, "tuning")

DEVELOPMENT = [
    "Kalunga2016",
    "BNCI2014_004",
    "BNCI2014_009",
    "BNCI2014_001",
    "BNCI2015_001",
]
CONFIRMATORY = [
    "MAMEM3",
    "Weibo2014",
    "BNCI2015_004",
    "BNCI2015_003",
    "Schirrmeister2017",
]
PARADIGM = {
    "BNCI2014_001": "MI",
    "BNCI2014_004": "MI",
    "BNCI2015_001": "MI",
    "BNCI2015_004": "MI",
    "Schirrmeister2017": "MI",
    "Weibo2014": "MI",
    "BNCI2014_009": "P300",
    "BNCI2015_003": "P300",
    "Kalunga2016": "SSVEP",
    "MAMEM3": "SSVEP",
}

# network -> (base class, architecture, learning rate); the learning rates are the benchmark's
NETWORKS = {
    "eegnet": ("torch", "EEGNet", 1e-3),
    "shallownet": ("torch", "ShallowNet", 1e-3),
    "green": ("torch", "GREEN", 3e-3),
    "eeginception_erp": ("braindecode", "EEGInceptionERP", 1e-3),
    "fbcnet": ("braindecode", "FBCNet", 1e-3),
    "atcnet": ("braindecode", "ATCNet", 1e-3),
}
# input: (sampling rate, band) applied when loading AND inside the network. "native" keeps the
# recording rate up to 250 Hz (MAMEM3 128 Hz, Weibo2014 200 Hz) and the band below 0.45 x rate
INPUTS = {"paper": (128.0, None), "native": (250.0, (0.5, 100.0))}
RECORDED_SFREQ = {"MAMEM3": 128.0, "Weibo2014": 200.0}
# training: optimiser, weight decay, label smoothing, batch, epochs
TRAINING = {
    "paper": ("adamw", 1e-2, 0.1, 32, 300),
    "published": ("adam", 0.0, 0.0, 64, 1000),
}
# in tie-break order: earlier (cheaper) configurations win ties
CONFIGS = {
    "paper": ("paper", "paper"),
    "paper_input_published_training": ("paper", "published"),
    "native_input_paper_training": ("native", "paper"),
    "native_input_published_training": ("native", "published"),
}
FROZEN_DEEP = {
    "eegnet",
    "shallownet",
    "green",
}  # in probabilities/; others in probabilities_extra/
MAIN_NETWORKS = [
    "eegnet",
    "shallownet",
    "green",
]  # the networks of the main ensemble (v1)


def make_voter(network, config, sfreq, smoke=False):
    from processing_eeg_methods.ensemble.extra_voters import BraindecodeVoter
    from processing_eeg_methods.ensemble.features import bandpass
    from processing_eeg_methods.ensemble.voters import TorchVoter

    kind, architecture, lr = NETWORKS[network]
    input_name, training_name = CONFIGS[config]
    low, high = INPUTS[input_name][1] or (1.0, 45.0)  # paper band = TorchVoter's
    band = (low, min(high, 0.45 * sfreq))
    optimiser, weight_decay, smoothing, batch_size, epochs = TRAINING[training_name]

    class TunedVoter(BraindecodeVoter if kind == "braindecode" else TorchVoter):
        def _prep(self, X):
            x = bandpass(X, band[0], band[1], self.sfreq)
            return ((x - x.mean(-1, keepdims=True)) / self.scale_).astype(np.float32)

        def fit(self, X, y):
            import torch
            from sklearn.preprocessing import LabelEncoder

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder_ = LabelEncoder().fit(y)
            self.classes_ = self.encoder_.classes_
            self.scale_ = (
                float(np.std(bandpass(X, band[0], band[1], self.sfreq))) + 1e-9
            )
            xt = torch.tensor(self._prep(X), device=device)
            yt = torch.tensor(
                self.encoder_.transform(y), dtype=torch.long, device=device
            )
            self.model_ = self._build(X.shape[1], X.shape[2], len(self.classes_)).to(
                device
            )
            if optimiser == "adamw":
                opt = torch.optim.AdamW(
                    self.model_.parameters(), lr=self.lr, weight_decay=weight_decay
                )
            else:
                opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
            for _ in range(self.epochs):
                self.model_.train()
                perm = torch.randperm(len(xt), device=device)
                for i in range(0, len(xt), self.batch_size):
                    idx = perm[i : i + self.batch_size]
                    if len(idx) < 2:
                        continue
                    opt.zero_grad()
                    loss_fn(self.model_(xt[idx]), yt[idx]).backward()
                    opt.step()
                sched.step()
            self.device_ = device
            del xt, yt
            return self

    return TunedVoter(
        architecture, sfreq, epochs=2 if smoke else epochs, lr=lr, batch_size=batch_size
    )


def load(name, input_name):
    from moabb import paradigms as P

    from processing_eeg_methods.ensemble.benchmark_moabb import get_dataset

    dataset, paradigm = get_dataset(name)
    if input_name == "native":
        sfreq, (fmin, fmax) = INPUTS["native"]
        sfreq = min(sfreq, RECORDED_SFREQ.get(name, sfreq))
        fmax = min(fmax, 0.45 * sfreq)
        kind = PARADIGM[name]
        kw = dict(fmin=fmin, fmax=fmax, resample=sfreq)
        if kind == "P300":
            paradigm = P.P300(**kw)
        elif kind == "SSVEP":
            paradigm = P.SSVEP(n_classes=paradigm.n_classes, **kw)
        elif type(paradigm).__name__ == "LeftRightImagery":
            paradigm = P.LeftRightImagery(**kw)
        else:
            paradigm = P.MotorImagery(n_classes=paradigm.n_classes, **kw)
    return dataset, paradigm


def frozen(name, subject, network):
    """The paper configuration's saved probabilities (oof, test, y_oof, y_test), or None."""
    if network in FROZEN_DEEP:
        path = os.path.join(BENCH, "probabilities", name, f"subject_{subject}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        return z[f"oof__{network}"], z[f"test__{network}"], z["y_oof"], z["y_test"]
    path = os.path.join(
        BENCH, "probabilities_extra", name, f"subject_{subject}__{network}.npz"
    )
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["oof"], z["test"], z["y_oof"], z["y_test"]


def tuned(name, subject, network, config):
    """Saved probabilities of one configuration (oof, test, y_oof, y_test), or None."""
    if config == "paper":
        return frozen(name, subject, network)
    path = os.path.join(
        OUT, "probabilities", name, f"subject_{subject}__{network}__{config}.npz"
    )
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["oof"], z["test"], z["y_oof"], z["y_test"]


def run(datasets, plan, smoke=False):
    """plan(dataset) -> list of (network, config) to train for that dataset."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder

    from processing_eeg_methods.ensemble.benchmark_extra import training_groups
    from processing_eeg_methods.ensemble.benchmark_moabb import split_sessions
    from processing_eeg_methods.ensemble.voting_ensemble import (
        _fit_one,
        _oof_one,
        _proba_full,
    )

    out = os.path.join(OUT, "smoke" if smoke else "probabilities")
    log = os.path.join(out, "runs.csv")
    for name in datasets:
        jobs = [(n, c) for n, c in plan(name) if c != "paper"]
        dataset, _ = load(name, "paper")
        subjects = dataset.subject_list[:1] if smoke else dataset.subject_list
        target = os.path.join(out, name)
        os.makedirs(target, exist_ok=True)
        for subject in subjects:
            reference = frozen(name, subject, "eegnet")
            if reference is None:
                continue  # excluded by the benchmark's rule (or never run): skip here too
            todo = [
                (n, c)
                for n, c in jobs
                if not os.path.exists(
                    os.path.join(target, f"subject_{subject}__{n}__{c}.npz")
                )
            ]
            for input_name in INPUTS:
                batch = [(n, c) for n, c in todo if CONFIGS[c][0] == input_name]
                if not batch:
                    continue
                ds, paradigm = load(name, input_name)
                X, y, meta = paradigm.get_data(ds, [subject])
                train, test, _ = split_sessions(meta)
                groups = training_groups(meta, train)
                encoder = LabelEncoder().fit(y[train])
                ytr, yte = encoder.transform(y[train]), encoder.transform(y[test])
                assert np.array_equal(ytr, reference[2]) and np.array_equal(
                    yte, reference[3]
                ), f"{name} subject {subject} ({input_name} input): trials differ from the benchmark's"
                Xtr, Xte, k = X[train], X[test], len(encoder.classes_)
                n_folds = min(5, len(np.unique(groups[train])))
                splits = list(GroupKFold(n_folds).split(Xtr, ytr, groups[train]))
                sfreq = paradigm.resample
                for network, config in batch:
                    t0 = time.perf_counter()
                    voter = make_voter(network, config, sfreq, smoke)
                    oof = _oof_one(voter, Xtr, ytr, splits, k)
                    final = _fit_one(
                        make_voter(network, config, sfreq, smoke), Xtr, ytr
                    )
                    proba = _proba_full(final, Xte, k)
                    np.savez_compressed(
                        os.path.join(
                            target, f"subject_{subject}__{network}__{config}.npz"
                        ),
                        oof=oof,
                        test=proba,
                        y_oof=ytr,
                        y_test=yte,
                    )
                    row = {
                        "dataset": name,
                        "subject": subject,
                        "network": network,
                        "config": config,
                        "seconds": round(time.perf_counter() - t0),
                        "oof_bacc": bacc(ytr, oof),
                        "finished": time.strftime("%Y-%m-%d %H:%M"),
                    }
                    pd.DataFrame([row]).to_csv(
                        log, mode="a", header=not os.path.exists(log), index=False
                    )
                    print(
                        f"{name} s{subject} {network:16s} {config:32s} {row['seconds']:6d}s "
                        f"oof {row['oof_bacc']:.3f}",
                        flush=True,
                    )


def bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, p.argmax(1)))


def choose(scores):
    """The selection rule. scores: one row per (paradigm, network, config, dataset, subject)
    with oof_bacc. Mean over subjects within a dataset, then over datasets; ties go to the
    earlier configuration in CONFIGS. Returns ({paradigm: {network: config}}, score table).
    """
    per_dataset = scores.groupby(
        ["paradigm", "network", "config", "dataset"]
    ).oof_bacc.mean()
    score = per_dataset.groupby(["paradigm", "network", "config"]).mean().reset_index()
    score["order"] = score.config.map({c: i for i, c in enumerate(CONFIGS)})
    best = (
        score.sort_values(["oof_bacc", "order"], ascending=[False, True])
        .groupby(["paradigm", "network"])
        .head(1)
    )
    selected = {p: dict(zip(g.network, g.config)) for p, g in best.groupby("paradigm")}
    return selected, score.drop(columns="order")


def select():
    """Development OOF only. Test-session results are not read."""
    rows = []
    for name in DEVELOPMENT:
        dataset, _ = load(name, "paper")
        for subject in dataset.subject_list:
            for network in NETWORKS:
                for config in CONFIGS:
                    f = tuned(name, subject, network, config)
                    if f is not None:
                        rows.append(
                            {
                                "dataset": name,
                                "paradigm": PARADIGM[name],
                                "subject": subject,
                                "network": network,
                                "config": config,
                                "oof_bacc": bacc(f[2], f[0]),
                            }
                        )
    df = pd.DataFrame(rows)
    counts = df.groupby(["dataset", "network"]).config.nunique()
    missing = counts[counts < len(CONFIGS)]
    assert missing.empty, f"tuning is incomplete:\n{missing}"
    selected, score = choose(df)
    score.to_csv(os.path.join(OUT, "development_scores.csv"), index=False)
    json.dump(selected, open(os.path.join(OUT, "selected.json"), "w"), indent=1)
    table = score.pivot_table(
        index=["paradigm", "network"], columns="config", values="oof_bacc"
    )
    print(table.round(3).to_string())
    print(json.dumps(selected, indent=1))


# ------------------------------------------------------------------------------------------
# Planned comparisons (confirmatory datasets, per subject, Wilcoxon signed-rank, rule auto):
#   1. main ensemble with tuned networks vs the voter picked on training data from the same pool
#   2. main ensemble with tuned networks vs each tuned network used alone
#   3. main ensemble with tuned networks vs the main ensemble with the benchmark's networks
# "Tuned" swaps the main ensemble's EEGNet, ShallowNet and GREEN for their selected configuration.


def subject_rows(name, selected):
    from processing_eeg_methods.ensemble.analyze_voters import (
        acc,
        best_single,
        ensemble_accuracy,
        load_subject,
    )

    rows = []
    folder = os.path.join(BENCH, "probabilities", name)
    for file in sorted(os.listdir(folder), key=lambda f: int(f[8:-4])):
        subject = int(file[8:-4])
        s = load_subject(os.path.join(folder, file))
        paper_pool = list(s["voters"])
        for network, config in selected.items():
            f = tuned(name, subject, network, config)
            if f is None:
                raise FileNotFoundError(
                    f"{name} subject {subject}: no {network} ({config})"
                )
            assert np.array_equal(f[2], s["y_oof"]) and np.array_equal(
                f[3], s["y_test"]
            )
            s["oof"][f"{network}_tuned"], s["test"][f"{network}_tuned"] = f[0], f[1]
            paper = frozen(name, subject, network)
            if paper is not None:
                s["test"][f"{network}_paper"] = paper[1]
        tuned_pool = [v for v in paper_pool if v not in MAIN_NETWORKS] + [
            f"{n}_tuned" for n in MAIN_NETWORKS
        ]
        row = {
            "dataset": name,
            "paradigm": PARADIGM[name],
            "subject": subject,
            "tuned ensemble": ensemble_accuracy(s, tuned_pool, "auto"),
            "paper ensemble": ensemble_accuracy(s, paper_pool, "auto"),
        }
        row["tuned pool: picked"], row["tuned pool: picked voter"] = best_single(
            s, tuned_pool
        )
        row["paper pool: picked"], _ = best_single(s, paper_pool)
        for network in NETWORKS:
            row[f"{network} (tuned) alone"] = acc(
                s["y_test"], s["test"][f"{network}_tuned"]
            )
            if f"{network}_paper" in s["test"]:
                row[f"{network} (paper) alone"] = acc(
                    s["y_test"], s["test"][f"{network}_paper"]
                )
        rows.append(row)
    return rows


def compare(df, a, b):
    from scipy.stats import wilcoxon

    # rounded so that equal accuracies tie exactly (float noise of ~1e-16 otherwise splits ties
    # and counts a tie as a win)
    d = np.round(100 * (df[a] - df[b]), 8)
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return {
        "comparison": f"{a} vs {b}",
        "mean A": 100 * df[a].mean(),
        "mean B": 100 * df[b].mean(),
        "diff (pp)": d.mean(),
        "A better": f"{(d > 0).sum()}/{len(d)}",
        "p": p,
    }


def table(df):
    rows = [compare(df, "tuned ensemble", "tuned pool: picked")]
    rows += [compare(df, "tuned ensemble", f"{n} (tuned) alone") for n in NETWORKS]
    rows += [
        compare(df, "tuned ensemble", "paper ensemble"),
        compare(df, "paper ensemble", "paper pool: picked"),
    ]
    rows += [
        compare(df, f"{n} (tuned) alone", f"{n} (paper) alone")
        for n in NETWORKS
        if f"{n} (paper) alone" in df
    ]
    t = pd.DataFrame(rows)
    return t.round({"mean A": 1, "mean B": 1, "diff (pp)": 1, "p": 4})


def analyze():
    selected = json.load(open(os.path.join(OUT, "selected.json")))
    lines = [
        "# Tuned networks: planned comparisons (Amendment A2)",
        "",
        "Per subject balanced accuracy on the test sessions; two-sided Wilcoxon signed-rank;",
        "ensemble rule auto. Main ensemble = the 12 v1 voters; tuned = its EEGNet, ShallowNet and",
        "GREEN replaced by the selected configuration (selected.json). Comparisons 1-3 are",
        "planned; the network-vs-network rows at the bottom are descriptive.",
        "",
    ]
    for title, datasets, tag in [
        ("Confirmatory (the test)", CONFIRMATORY, "confirmatory"),
        ("Development (configurations chosen here)", DEVELOPMENT, "development"),
    ]:
        df = pd.DataFrame(
            [
                r
                for name in datasets
                for r in subject_rows(name, selected[PARADIGM[name]])
            ]
        )
        df.to_csv(os.path.join(OUT, f"results_{tag}.csv"), index=False)
        per_dataset = df.groupby("dataset")[
            [
                "tuned ensemble",
                "paper ensemble",
                "tuned pool: picked",
                "atcnet (tuned) alone",
            ]
        ].mean()
        picked = df["tuned pool: picked voter"].value_counts()
        lines += [
            f"## {title}: {len(df)} subjects",
            "",
            table(df).to_markdown(index=False),
            "",
            "Per dataset (mean %):",
            "",
            (100 * per_dataset).round(1).to_markdown(),
            "",
            "Voter picked on training data (tuned pool): "
            + ", ".join(f"{k} {v}" for k, v in picked.items()),
            "",
        ]
    open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


def main(argv=None):
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    argv = sys.argv[1:] if argv is None else argv
    smoke = "--smoke" in argv
    command = argv[0]
    if command == "tune":
        datasets = DEVELOPMENT[:1] if smoke else DEVELOPMENT
        run(datasets, lambda name: [(n, c) for n in NETWORKS for c in CONFIGS], smoke)
    elif command == "select":
        select()
    elif command == "confirm":
        selected = json.load(open(os.path.join(OUT, "selected.json")))
        run(CONFIRMATORY, lambda name: list(selected[PARADIGM[name]].items()), smoke)
    elif command == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
