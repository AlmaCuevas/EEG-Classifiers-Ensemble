"""Amendment A3 (PROTOCOL.md): the standard pipeline of each paradigm, as extra voters.

  MI (motor imagery, mental imagery, motor execution)   fbcsp       filter-bank CSP + LDA
  P300                                                   xdawn_lda   xDAWN + shrinkage LDA
  SSVEP                                                  cca, trca   CCA and ensemble TRCA

Each is added only to the datasets of its paradigm. Settings follow the original papers and
were fixed before any of these voters was run; nothing is tuned.

  python -m processing_eeg_methods.ensemble.standard_voters run [--datasets ...]
      same trials, chronological split and grouped 5-fold OOF as the benchmark
      -> Results/ensemble_benchmark/probabilities_standard/<dataset>/subject_<id>__<voter>.npz
  python -m processing_eeg_methods.ensemble.standard_voters analyze
      the planned comparisons -> Results/ensemble_benchmark/standard/RESULTS.md
"""

import argparse
import os
import sys
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from processing_eeg_methods.ensemble.features import bandpass
from processing_eeg_methods.ensemble.tune_networks import (
    BENCH,
    CONFIRMATORY,
    DEVELOPMENT,
    PARADIGM,
)
from processing_eeg_methods.ensemble.voters import _lr

OUT = os.path.join(BENCH, "probabilities_standard")
RESULTS = os.path.join(BENCH, "standard")
STANDARD = {"MI": ["fbcsp"], "P300": ["xdawn_lda"], "SSVEP": ["cca", "trca"]}


def _shrinkage_lda(priors=None):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Log-variance of 4 CSP components in each 4-Hz band from 4 to 40 Hz (Ang et al. 2008)."""

    def __init__(self, sfreq=128.0, n_components=4):
        self.sfreq, self.n_components = sfreq, n_components

    def _bands(self):
        return [(lo, lo + 4) for lo in range(4, 40, 4) if lo + 4 <= 0.45 * self.sfreq]

    def fit(self, X, y):
        from mne.decoding import CSP

        self.csps_ = [
            CSP(self.n_components, reg="ledoit_wolf", log=True).fit(
                bandpass(X, lo, hi, self.sfreq), y
            )
            for lo, hi in self._bands()
        ]
        return self

    def transform(self, X):
        return np.concatenate(
            [
                csp.transform(bandpass(X, lo, hi, self.sfreq))
                for csp, (lo, hi) in zip(self.csps_, self._bands())
            ],
            axis=1,
        )


class BinMeans(BaseEstimator, TransformerMixin):
    """Flattened time course, averaged in bins of `width` samples (decimation for ERPs)."""

    def __init__(self, width=4):
        self.width = width

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = X.shape[-1] // self.width * self.width
        binned = X[..., :n].reshape(*X.shape[:-1], -1, self.width).mean(-1)
        return binned.reshape(len(X), -1)


class CCAFeatures(BaseEstimator, TransformerMixin):
    """Largest canonical correlation between the trial and sine/cosine references at each
    stimulus frequency and its harmonics (Lin et al. 2006). Training-free."""

    def __init__(self, sfreq=128.0, frequencies=(), n_harmonics=2):
        self.sfreq, self.frequencies, self.n_harmonics = sfreq, frequencies, n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.arange(X.shape[-1]) / self.sfreq
        refs = []
        for f in self.frequencies:
            ref = [
                fn(2 * np.pi * h * f * t)
                for h in range(1, self.n_harmonics + 1)
                for fn in (np.sin, np.cos)
            ]
            ref = np.array(ref).T
            refs.append(np.linalg.qr(ref - ref.mean(0))[0])
        out = np.empty((len(X), len(refs)))
        for i, trial in enumerate(X):
            qx = np.linalg.qr((trial - trial.mean(-1, keepdims=True)).T)[0]
            for j, qy in enumerate(refs):
                out[i, j] = np.linalg.svd(qx.T @ qy, compute_uv=False)[0]
        return out


class TRCAFeatures(BaseEstimator, TransformerMixin):
    """Ensemble task-related component analysis (Nakanishi et al. 2018): one spatial filter
    per class maximising the reproducibility of its trials; a trial's feature for class k is
    the correlation between its filtered signal and the filtered class-k template."""

    def fit(self, X, y):
        from scipy.linalg import eigh

        x = X - X.mean(-1, keepdims=True)
        self.classes_ = np.unique(y)
        filters, self.templates_ = [], []
        for k in self.classes_:
            xk = x[y == k]
            total = xk.sum(0)
            s = total @ total.T - np.einsum("nct,ndt->cd", xk, xk)
            q = np.einsum("nct,ndt->cd", xk, xk)
            q += 1e-6 * np.trace(q) / len(q) * np.eye(len(q))
            filters.append(eigh(s, q)[1][:, -1])
            self.templates_.append(xk.mean(0))
        self.w_ = np.array(filters).T
        return self

    def transform(self, X):
        x = X - X.mean(-1, keepdims=True)
        projected = np.einsum("ck,nct->nkt", self.w_, x)
        out = np.empty((len(X), len(self.templates_)))
        for j, template in enumerate(self.templates_):
            ref = (self.w_.T @ template).ravel()
            ref = (ref - ref.mean()) / (ref.std() + 1e-12)
            flat = projected.reshape(len(X), -1)
            flat = (flat - flat.mean(1, keepdims=True)) / (
                flat.std(1, keepdims=True) + 1e-12
            )
            out[:, j] = flat @ ref / len(ref)
        return out


def frequencies(class_names):
    """Stimulus frequency of each class name ('13', '6.66'); classes such as 'rest' have none."""
    out = []
    for c in class_names:
        try:
            out.append(float(c))
        except ValueError:
            pass
    return tuple(out)


def standard_voters(paradigm, sfreq=128.0, class_names=()):
    """The standard voters of one paradigm ("MI", "P300" or "SSVEP")."""
    from pyriemann.spatialfilters import Xdawn

    voters = {
        "fbcsp": make_pipeline(
            FilterBankCSP(sfreq),
            SelectKBest(partial(mutual_info_classif, random_state=42), k=8),
            _shrinkage_lda(),
        ),
        "xdawn_lda": make_pipeline(
            # equal priors: P300 data are ~5:1 non-target, and the score is balanced accuracy
            Xdawn(nfilter=4, estimator="oas"),
            BinMeans(4),
            _shrinkage_lda(priors=np.array([0.5, 0.5])),
        ),
        "cca": make_pipeline(
            CCAFeatures(sfreq, frequencies(class_names)), StandardScaler(), _lr(0.1)
        ),
        "trca": make_pipeline(TRCAFeatures(), StandardScaler(), _lr(0.1)),
    }
    return {name: clone(voters[name]) for name in STANDARD[paradigm]}


def frozen_labels(name, subject):
    path = os.path.join(BENCH, "probabilities", name, f"subject_{subject}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["y_oof"], z["y_test"]


def run(datasets):
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder

    from processing_eeg_methods.ensemble.benchmark_extra import training_groups
    from processing_eeg_methods.ensemble.benchmark_moabb import (
        SFREQ,
        get_dataset,
        split_sessions,
    )
    from processing_eeg_methods.ensemble.voting_ensemble import (
        _fit_one,
        _oof_one,
        _proba_full,
    )

    timing_csv = os.path.join(OUT, "timing.csv")
    for name in datasets:
        dataset, paradigm = get_dataset(name)
        target = os.path.join(OUT, name)
        os.makedirs(target, exist_ok=True)
        for subject in dataset.subject_list:
            reference = frozen_labels(name, subject)
            if reference is None:
                continue  # excluded by the benchmark's rule
            names = STANDARD[PARADIGM[name]]
            todo = [
                v
                for v in names
                if not os.path.exists(
                    os.path.join(target, f"subject_{subject}__{v}.npz")
                )
            ]
            if not todo:
                continue
            X, y, meta = paradigm.get_data(dataset, [subject])
            train, test, _ = split_sessions(meta)
            groups = training_groups(meta, train)
            encoder = LabelEncoder().fit(y[train])
            ytr, yte = encoder.transform(y[train]), encoder.transform(y[test])
            assert np.array_equal(ytr, reference[0]) and np.array_equal(
                yte, reference[1]
            ), f"{name} subject {subject}: trials differ from the benchmark's"
            Xtr, Xte, k = X[train], X[test], len(encoder.classes_)
            n_folds = min(5, len(np.unique(groups[train])))
            splits = list(GroupKFold(n_folds).split(Xtr, ytr, groups[train]))
            pool = standard_voters(PARADIGM[name], SFREQ, encoder.classes_)
            for v in todo:
                t0 = time.perf_counter()
                oof = _oof_one(pool[v], Xtr, ytr, splits, k)
                t1 = time.perf_counter()
                model = _fit_one(pool[v], Xtr, ytr)
                t2 = time.perf_counter()
                proba = _proba_full(model, Xte, k)
                latencies = []
                for i in range(min(30, len(Xte))):  # one trial at a time, as online
                    s = time.perf_counter()
                    _proba_full(model, Xte[i : i + 1], k)
                    latencies.append(time.perf_counter() - s)
                row = {
                    "dataset": name,
                    "subject": subject,
                    "voter": v,
                    "oof_seconds": t1 - t0,
                    "fit_seconds": t2 - t1,
                    "latency_ms": 1000 * float(np.median(latencies[1:] or latencies)),
                    "n_train": len(ytr),
                    "n_channels": X.shape[1],
                    "n_times": X.shape[2],
                    "epoch_seconds": X.shape[2] / SFREQ,
                }
                np.savez_compressed(
                    os.path.join(target, f"subject_{subject}__{v}.npz"),
                    test=proba,
                    oof=oof,
                    y_test=yte,
                    y_oof=ytr,
                )
                pd.DataFrame([row]).to_csv(
                    timing_csv,
                    mode="a",
                    header=not os.path.exists(timing_csv),
                    index=False,
                )
                print(
                    f"{name} s{subject} {v:10s} oof {row['oof_seconds']:7.1f}s "
                    f"fit {row['fit_seconds']:6.1f}s latency {row['latency_ms']:6.1f}ms",
                    flush=True,
                )


# ------------------------------------------------------------------------------------------
# Planned comparisons (PROTOCOL.md, Amendment A3). "Full" pool = the 12 v1 voters plus the
# standard voters of the dataset's paradigm.


def subject_rows(name):
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
        v1 = list(s["voters"])
        standard = STANDARD[PARADIGM[name]]
        for v in standard:
            z = np.load(os.path.join(OUT, name, f"subject_{subject}__{v}.npz"))
            assert np.array_equal(z["y_oof"], s["y_oof"]) and np.array_equal(
                z["y_test"], s["y_test"]
            )
            s["oof"][v], s["test"][v] = z["oof"], z["test"]
        full = v1 + standard
        row = {
            "dataset": name,
            "paradigm": PARADIGM[name],
            "subject": subject,
            "full ensemble": ensemble_accuracy(s, full, "auto"),
            "v1 ensemble": ensemble_accuracy(s, v1, "auto"),
        }
        row["full pool: picked"], row["full pool: picked voter"] = best_single(s, full)
        row["v1 pool: picked"], _ = best_single(s, v1)
        for v in standard:
            row[f"{v} alone"] = acc(s["y_test"], s["test"][v])
        rows.append(row)
    return rows


def test(df, a, b, alternative="two-sided"):
    from scipy.stats import wilcoxon

    d = np.round(100 * (df[a] - df[b]), 8)  # equal accuracies tie exactly
    p = wilcoxon(d, alternative=alternative).pvalue if (d != 0).any() else 1.0
    return {
        "comparison": f"{a} vs {b}",
        "n": len(d),
        "mean A": 100 * df[a].mean(),
        "mean B": 100 * df[b].mean(),
        "diff (pp)": d.mean(),
        "A better": f"{(d > 0).sum()}/{len(d)}",
        "p": p,
    }


def holm(p):
    order, running, out = np.argsort(p), 0.0, np.empty(len(p))
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        out[j] = running
    return out


def planned(df):
    """H6 (one-sided) and the secondary comparisons (two-sided, Holm over the secondary set)."""
    primary = test(df, "full ensemble", "full pool: picked", "greater")
    secondary = [test(df, "full ensemble", "v1 ensemble")]
    for paradigm, voters in STANDARD.items():
        g = df[df.paradigm == paradigm]
        if len(g):
            secondary += [test(g, "full ensemble", f"{v} alone") for v in voters]
    t = pd.DataFrame([primary] + secondary)
    t["Holm p"] = [np.nan] + list(holm(t.p.to_numpy()[1:]))
    return t.round({"mean A": 1, "mean B": 1, "diff (pp)": 1, "p": 4, "Holm p": 4})


def analyze():
    os.makedirs(RESULTS, exist_ok=True)
    lines = [
        "# Paradigm-standard voters: planned comparisons (Amendment A3)",
        "",
        "Full pool = the 12 v1 voters plus FBCSP (MI), xDAWN + LDA (P300) or CCA and TRCA (SSVEP).",
        "Balanced accuracy on the test sessions; ensemble rule auto. First row: H6, one-sided",
        "Wilcoxon. Other rows: two-sided Wilcoxon, Holm-adjusted over those rows.",
        "",
    ]
    for title, datasets, tag in [
        ("Confirmatory (the test)", CONFIRMATORY, "confirmatory"),
        ("Development", DEVELOPMENT, "development"),
    ]:
        df = pd.DataFrame([r for name in datasets for r in subject_rows(name)])
        df.to_csv(os.path.join(RESULTS, f"results_{tag}.csv"), index=False)
        cols = ["full ensemble", "v1 ensemble", "full pool: picked", "v1 pool: picked"]
        cols += [c for c in df if c.endswith(" alone")]
        picked = df["full pool: picked voter"].value_counts()
        lines += [
            f"## {title}: {len(df)} subjects",
            "",
            planned(df).to_markdown(index=False),
            "",
            "Per dataset (mean %):",
            "",
            (100 * df.groupby("dataset")[cols].mean()).round(1).to_markdown(),
            "",
            "Voter picked on training data (full pool): "
            + ", ".join(f"{k} {v}" for k, v in picked.items()),
            "",
        ]
    open(os.path.join(RESULTS, "RESULTS.md"), "w", encoding="utf-8").write(
        "\n".join(lines)
    )
    print("\n".join(lines))


def main(argv=None):
    import mne

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["run", "analyze"])
    parser.add_argument("--datasets", nargs="+", default=DEVELOPMENT + CONFIRMATORY)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.step == "run":
        run(args.datasets)
    else:
        analyze()


if __name__ == "__main__":
    main()
