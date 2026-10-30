# EEG-Classifiers-Ensemble

Leakage-safe voting ensembles of diverse EEG classifiers ("voters"), plus the pre-registered
cross-session benchmark that tests whether they help.

A voting ensemble beats its members only when each voter is better than chance and the voters
make *different* mistakes (Condorcet's jury theorem). This package is built around that idea:
- **Diverse voters,** each reading a different aspect of the signal: Riemannian covariance,
  ERP time course, spectrum, coherence, complexity, random kernels and neural networks.
- **Out-of-fold weights.** Each voter is weighted by its accuracy on training trials it did
  not see, with folds grouped by run so that neighbouring trials cannot leak into the estimate.
- **An honest baseline.** The ensemble is compared with the single voter you would have
  picked from the same training data.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/AlmaCuevas/EEG-Classifiers-Ensemble.git
cd EEG-Classifiers-Ensemble
python -m venv .venv
# activate it: .venv\Scripts\activate (Windows) or source .venv/bin/activate
pip install -e ".[features]"        # classical voters (no GPU needed)
pip install -e ".[all]"             # + network voters and the MOABB benchmark datasets
python tools/install_green.py       # + the GREEN network voter (fetched from Roche/neuro-green)
```

For GPU training, install the CUDA build of PyTorch first ([pytorch.org](https://pytorch.org)).

## Use it on your data

### From Python

`X` holds the epoched trials as `(trials, channels, samples)`. `y` holds the labels. Pass the run
(or block, or session) of every trial as `groups`, so that the out-of-fold weights are not
inflated by neighbouring, correlated trials.

```python
from processing_eeg_methods.ensemble import VotingEnsemble
from processing_eeg_methods.ensemble.voter_sets import get_voters

ens = VotingEnsemble(get_voters("fast", sfreq=128.0))   # or "v1", "v2", or a list of names
ens.fit(X_train, y_train, groups=run_of_each_trial)
proba = ens.predict_proba(X_new)
print(ens.diversity_report())     # out-of-fold accuracy, weight and error correlation per voter
```

| Voter set | Voters | What it is |
|---|---|---|
| `fast` | 10 | The classical voters of v2. No GPU; calibrates in seconds to minutes. |
| `v1` | 12 | The pre-registered set, including EEGNet, ShallowNet and GREEN. |
| `v2` | 15 | Chosen from 20 voters on the development data (see the results below). |

You can also pass your own scikit-learn classifiers:
`VotingEnsemble({"name": classifier, ...})`.

How the votes are combined (`combination=`):
- **`"auto"` (default):** stacking when there are at least 40 training trials per class,
  weighted voting otherwise.
- **`"weighted"`, `"stacking"`, `"soft"`, `"majority"`:** fixed rules. Simple averaging
  (`soft`, `majority`) fell below the best single voter in the benchmark.

`predict_accumulated()` combines several repetitions of the same intended command into one
decision.

### From saved files, with the paper's protocol

Save one file per subject in `Datasets/<name>/`:

| File | Contents |
|---|---|
| `subject_<id>.npz` | `X` (trials, channels, samples), `y` (labels), `sfreq`; optional `session` and `run` per trial |
| `subject_<id>-epo.fif` | MNE Epochs. Labels come from the event names; `session` and `run` from `epochs.metadata` if present. |

Trials must be in recording order. Then run:

```bash
python -m processing_eeg_methods.ensemble.run_own_data <name> --voters fast
```

This runs exactly the benchmark's procedure:
1. Train on the first session and test on the later ones. With a single session, train on the
   first half of the runs (or trials) and test on the second half.
2. Estimate out-of-fold accuracy with folds grouped by run, inside the training data only.
3. Report every combination rule against the best single voter picked on training data.

The data are band-pass filtered (1–45 Hz; use `--no-filter` if you filtered the continuous
recording) and resampled to 128 Hz. Per-subject probabilities and `per_subject.csv` go to
`Results/own_data/<name>/`. `Datasets/` and `Results/` are never committed.

## Replicate the paper

The benchmark uses 10 public [MOABB](https://moabb.neurotechx.com) datasets. MOABB downloads
them on first use (24 GB) to `MNE_DATA` (default `~/mne_data`).

```bash
python -m processing_eeg_methods.ensemble.reproduce check      # code, environment, GPU, GREEN
python -m processing_eeg_methods.ensemble.reproduce all        # benchmark, timing, analysis
```

The steps can also run one at a time: `benchmark`, `timing`, then `analyze`. Use
`--datasets BNCI2014_001` for a single dataset.
- **Resumable:** every step skips subjects that are already finished.
- **Output:** results are written to `Results/ensemble_benchmark/`.
- **Compute:** about 120 GPU-hours on one 8 GB laptop GPU. Most of that is Schirrmeister2017,
  which also needs 16 GB of RAM with nothing else running.

**Exact environment.** The paper ran on Python 3.10, Windows and CUDA 12.1, with the versions
in [requirements-ensemble.lock.txt](requirements-ensemble.lock.txt):

```bash
py -3.10 -m venv ensemble_env
ensemble_env\Scripts\python -m pip install -r requirements-ensemble.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
ensemble_env\Scripts\python -m pip install -e . --no-deps
```

The lock file is an archival record of the paper's environment, and it is part of the
pre-registered fingerprint. Several of its pinned packages have published security advisories:
Pillow, urllib3, filelock, Lightning and PyTorch 2.5.1. Use it only in an isolated environment
to replicate the paper. For anything else, install with `pip install -e ".[...]"` as above;
that installation passed `pip-audit`.

**Verified on 2026-10-30.** Each environment was checked from a fresh clone on Kalunga2016,
12 subjects:
- **The paper's exact environment** reproduced every voter and every combination rule
  bit-for-bit, neural networks included.
- **Python 3.14 with the newest libraries** (numpy 2.5, pandas 3.0, scikit-learn 1.9, PyTorch
  2.14):
  - all 16 tests pass;
  - the 13 classical voters were bit-identical;
  - the network voters and the SVM voter differed subject by subject, because the random number
    streams changed. On this small dataset the ensemble's mean moved by 0.7 points.
- **Newer versions are therefore not exact replication.** We have not re-verified the paper's
  conclusions under them; use the exact environment to replicate. `reproduce check` lists every
  package that differs from the paper's.
- **Version caps.** `pyproject.toml` caps braindecode below 1.6 and pyRiemann below 0.14,
  because later releases remove names the frozen code uses.

**Integrity.**
- [PROTOCOL.md](src/processing_eeg_methods/ensemble/PROTOCOL.md) holds the pre-registration:
  hypotheses, datasets, exclusion rules, every deviation, and SHA-256 fingerprints of the code
  that produced the results.
- `reproduce check` verifies those fingerprints.

## Results

The pre-registered cross-session benchmark:
- **Split:** train on the first session, test on later sessions.
- **Voters:** v1.
- **Combination:** `auto`.
- **Baseline:** the single voter picked on training data.
- **Metric:** balanced accuracy, mean over subjects. Gain is in points, with the number of
  subjects improved in brackets.

| Dataset | Paradigm | Subjects | Best single voter | Ensemble | Gain |
|---|---|---|---|---|---|
| *Confirmatory (not used to design anything)* | | | | | |
| BNCI2015_004 | Mental imagery, 5-class | 9 | 34.5% | 36.9% | +2.4 (6/9) |
| Schirrmeister2017 | Motor execution, 4-class | 14 | 89.2% | 92.3% | +3.1 (13/14) |
| Weibo2014 | Motor imagery, 2-class | 10 | 70.6% | 73.2% | +2.7 (6/10) |
| BNCI2015_003 | P300 | 10 | 73.4% | 75.2% | +1.8 (6/10) |
| MAMEM3 | SSVEP, 5-class | 10 | 26.0% | 27.7% | +1.7 (5/10) |
| **All confirmatory** | | **53** | **61.5%** | **63.9%** | **+2.4 ± 5.5 (36/53), p = 0.0008** |
| *Development (used to design the method)* | | | | | |
| BNCI2014_001 | Motor imagery, 4-class | 9 | 72.0% | 76.5% | +4.5 (8/9) |
| BNCI2014_004 | Motor imagery, 2-class | 9 | 70.6% | 74.4% | +3.7 (6/9) |
| BNCI2015_001 | Motor imagery, 2-class | 12 | 78.3% | 84.6% | +6.4 (11/12) |
| BNCI2014_009 | P300 | 10 | 79.8% | 84.3% | +4.5 (10/10) |
| Kalunga2016 | SSVEP, 4-class | 12 | 52.0% | 53.6% | +1.6 (6/12) |

The full set of findings, with practical guidance, is in [FINDINGS.md](FINDINGS.md). It covers:
- why picking one voter loses to voting;
- why voters must be above chance rather than good;
- comparisons with published results.

What the benchmark shows:
- **Where the voters carry signal, the ensemble beats the voter you would have picked,** on
  every confirmatory dataset. When every voter is near chance it cannot help (Condorcet).
- **The gain comes from hedging an unreliable choice.**
  - On the confirmatory data, the best voter in hindsight was 4.0 points above the one
    picked from training data.
  - The ensemble recovered 60% of that gap.
  - The error correlation between voters did not predict the gain.
- **Don't curate the voters on small data.**
  - v2 (15 of 20 voters, picked on development data) looked 2.6 points better there.
  - On the confirmatory data it was 0.2 points worse than v1 (p = 0.71).
- **Fast and online use.**
  - Prediction is fast: the `fast` set needs a median of 34 ms per trial, and the 15-voter
    `v2` set 52 ms, run serially on a laptop. That fits an online loop.
  - Calibration is the bottleneck, and it is dominated by the networks.
  - The `fast` set needs no GPU and still beats its own best voter (+1.7 points, p = 0.002).
    It is 1.6 points below the full ensemble.
  - ATCNet was the most expensive voter, and removing it changed nothing measurable.

Datasets that could not be evaluated without leakage:
- **Zhou2016:** unreadable with current MNE.
- **Nakanishi2015:** a single run, so a chronological split leaves classes out of training.
- **Cho2017:** trials are stored grouped by class, so the recording order is lost.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Cite

See [CITATION.cff](CITATION.cff). The paper is in preparation.

## Acknowledgments

Built on:
- [MOABB](https://github.com/NeuroTechX/moabb) for datasets;
- [pyRiemann](https://github.com/pyRiemann/pyRiemann) for Riemannian geometry;
- [braindecode](https://github.com/braindecode/braindecode) for EEGNet, ShallowNet, ATCNet,
  FBCNet and EEGInception;
- [GREEN](https://github.com/Roche/neuro-green) (Paillard et al., 2025, Apache-2.0);
- [sktime](https://github.com/sktime/sktime) for MiniRocket;
- [antropy](https://github.com/raphaelvallat/antropy).

## License

MIT
