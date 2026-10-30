"""Exploratory (written 2026-10-30, after the confirmatory results were known; see PROTOCOL.md):
could one voter, chosen in advance, replace the ensemble?

From the saved v1 results (nothing is retrained), on every subject:
  - which voter was truly best on the test data (hindsight);
  - how often the voter picked on the subject's own training data was that voter;
  - the ensemble against every way of choosing one voter: picked on own training data,
    best on the development datasets overall, best on development datasets of the same
    paradigm, and the hindsight best (not deployable);
  - the ensemble's gain against how far the picked voter is above chance.

Writes Results/ensemble_benchmark/analysis/selection_strategies.csv, hindsight_winners.csv,
gain_vs_chance.csv and selection_summary.json.

Usage: python -m processing_eeg_methods.ensemble.explore_selection
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from processing_eeg_methods.ensemble.benchmark_moabb import OUT_DIR
from processing_eeg_methods.ensemble.select_voters import CONFIRMATORY, DEVELOPMENT

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
CHANCE_BINS = [-1, 0.05, 0.15, 0.30, 1]
CHANCE_LABELS = ["< 5 pp", "5-15 pp", "15-30 pp", "> 30 pp"]


def load():
    frames = []
    for d in DEVELOPMENT + CONFIRMATORY:
        t = pd.read_csv(os.path.join(OUT_DIR, f"{d}.csv"))
        z = np.load(
            os.path.join(
                OUT_DIR, "probabilities", d, f"subject_{t.subject.iloc[0]}.npz"
            )
        )
        t["dataset"], t["chance"] = d, 1 / len(z["classes"])
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    ps = pd.read_csv(os.path.join(OUT_DIR, "analysis", "per_subject.csv"))
    t = t.merge(
        ps[["dataset", "subject", "auto", "best_single", "oracle_best_voter"]],
        on=["dataset", "subject"],
    )
    t["set"] = np.where(t.dataset.isin(CONFIRMATORY), "Confirmatory", "Development")
    return t


def compare(name, ensemble, single):
    gain = ensemble - single
    return {
        "strategy": name,
        "accuracy": single.mean(),
        "ensemble": ensemble.mean(),
        "ensemble_minus": gain.mean(),
        "ensemble_minus_sd": gain.std(),
        "ensemble_better": int((gain > 0).sum()),
        "ties": int((gain == 0).sum()),
        "n": len(gain),
        "p_two_sided": wilcoxon(gain).pvalue if (gain != 0).any() else 1.0,
    }


def main():
    t = load()
    voters = [c for c in t if c.startswith("voter: ")]
    t["hindsight_voter"] = t[voters].idxmax(axis=1).str[len("voter: ") :]
    dev, conf = t[t.set == "Development"], t[t.set == "Confirmatory"]
    overall = dev[voters].mean().idxmax()
    by_paradigm = {
        p: dev[dev.dataset.map(PARADIGM) == p][voters].mean().idxmax()
        for p in ["MI", "P300", "SSVEP"]
    }
    same_paradigm = pd.Series(
        [r[by_paradigm[PARADIGM[r.dataset]]] for _, r in conf.iterrows()],
        index=conf.index,
    )
    rows = [
        compare(
            "picked on the subject's own training data (baseline)",
            conf["auto"],
            conf.best_single,
        ),
        compare(
            f"best on development data, same paradigm ({', '.join(f'{p}: {v[7:]}' for p, v in by_paradigm.items())})",
            conf["auto"],
            same_paradigm,
        ),
        compare(
            f"best on development data overall ({overall[7:]})",
            conf["auto"],
            conf[overall],
        ),
        compare(
            "hindsight best per subject (not deployable)",
            conf["auto"],
            conf.oracle_best_voter,
        ),
    ]
    strategies = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "analysis")
    strategies.to_csv(os.path.join(out, "selection_strategies.csv"), index=False)

    winners = (
        t.groupby(["set", "hindsight_voter"])
        .size()
        .rename("subjects")
        .reset_index()
        .sort_values(["set", "subjects"], ascending=[True, False])
    )
    winners.to_csv(os.path.join(out, "hindsight_winners.csv"), index=False)
    picked_is_best = conf.picked_voter == conf.hindsight_voter

    t["above_chance"] = t.best_single - t.chance
    t["gain"] = t["auto"] - t.best_single
    t["bin"] = pd.cut(t.above_chance, CHANCE_BINS, labels=CHANCE_LABELS)
    bins = (
        t.groupby("bin", observed=True)
        .gain.agg(
            subjects="size", mean_gain="mean", improved=lambda g: int((g > 0).sum())
        )
        .reset_index()
    )
    bins.to_csv(os.path.join(out, "gain_vs_chance.csv"), index=False)
    rho = spearmanr(t.above_chance, t.gain)
    json.dump(
        {
            "confirmatory_subjects": len(conf),
            "picked_is_hindsight_best": int(picked_is_best.sum()),
            "subjects": len(t),
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
        },
        open(os.path.join(out, "selection_summary.json"), "w"),
        indent=1,
    )

    pd.set_option("display.width", 200)
    print(
        f"Confirmatory: the picked voter was the hindsight best for {picked_is_best.sum()}/{len(conf)} subjects; "
        f"{conf.hindsight_voter.nunique()} different voters were the hindsight best for at least one subject."
    )
    print(strategies.round(4).to_string(index=False))
    print(
        f"\nGain vs distance of the picked voter above chance (all {len(t)} subjects): "
        f"Spearman rho = {rho.statistic:.2f}, p = {rho.pvalue:.2f}"
    )
    print(bins.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
