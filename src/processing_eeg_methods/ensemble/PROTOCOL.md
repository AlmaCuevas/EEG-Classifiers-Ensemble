# Pre-registered protocol: does a diverse voting ensemble beat its best voter?

Frozen on **2026-10-26**, before any confirmatory dataset was analysed. Changes after
this date must be listed under "Deviations" with a reason.

**Names in the paper** (added 2026-11-18; nothing below changes): the main ensemble is v1,
the curated ensemble v2 and the classical ensemble `fast`; experiment 3 (tuned networks) is
Amendment A2 and experiment 4 (paradigm-standard voters) is Amendment A3.

## Question
Across EEG paradigms, does combining many *diverse* classifiers ("voters") beat the
single best voter that one would have chosen from the same training data? And does
the size of the gain depend on how diverse (uncorrelated) the voters' errors are?

## Method (frozen)
- **Code:** `processing_eeg_methods.ensemble`, at the commit that adds this file.
- **Environment:** `requirements-ensemble.lock.txt`.
- **Voters (12):** `default_voters()` plus `deep_voters(epochs=300)`:
  - covariance tangent space (6 bands);
  - covariance tangent space (4–30 Hz);
  - ERP time course;
  - relative spectrum;
  - coherence;
  - Riemannian MDM 8–30 Hz;
  - complexity + gradient boosting;
  - MiniRocket;
  - CSP+LDA 8–30 Hz;
  - EEGNet;
  - ShallowNet;
  - GREEN.
- **Combination:** `combination="auto"`. This is stacking (logistic regression on
  out-of-fold log-probabilities) when the training data has at least 40 trials per class;
  otherwise votes are weighted by out-of-fold balanced accuracy above chance.
- **Out-of-fold estimates:** 5-fold GroupKFold inside the training data, grouped by run
  (or contiguous blocks when there are fewer than 3 runs).
- **Preprocessing:** MOABB paradigm, 1–45 Hz (1–24 Hz for P300), resampled to 128 Hz.
- **Evaluation split:**
  - Train on the first session; test on all later sessions.
  - With one session: train on the first half of the runs, test on the second half.
  - The test data is never used for any choice.
- **Baseline:** the single voter with the highest out-of-fold balanced accuracy on the
  training data, i.e. what one would deploy without an ensemble.
- **Metric:** balanced accuracy on the test data, per subject.

## Hypotheses and tests (confirmatory datasets only)
- **H1.** The ensemble beats the baseline. Test: one-sided Wilcoxon signed-rank test on
  per-subject differences, pooled over all confirmatory subjects, α = 0.05. Also reported
  per dataset, with the number of subjects improved.
- **H2.** The gain is larger when voters' errors are less correlated. Test: Spearman
  correlation between per-subject out-of-fold error correlation and gain; one-sided,
  negative.
- **H3.** Accuracy grows with the number of voters. Shown as gain vs k for random voter
  subsets; descriptive only.
- **Exploratory, not confirmatory:** leave-one-voter-out, classical vs deep families, and
  the comparison of combination rules.

## Datasets
- **Development, used to design the method:** BNCI2014_001, BNCI2014_004, BNCI2015_001,
  BNCI2014_009, Kalunga2016, plus the imagined-speech datasets (BrainCommand, IC BCI 2020
  Track 3, Aguilera, Coretto).
- **Confirmatory:** fixed below after the loading check. That check verifies only that a
  dataset downloads, loads, keeps its recording order and has a chronological split. No
  accuracy is looked at.

### Confirmatory set (fixed 2026-10-26, before any of these datasets was analysed)

| Dataset | Paradigm | Subjects | Split |
|---|---|---|---|
| BNCI2015_004 | 5-class mental imagery | 9 | session 1 → session 2 |
| Schirrmeister2017 (HGD) | 4-class motor execution | 14 | official train run → test run |
| Weibo2014 | 2-class motor imagery | 10 | first half → second half (single run) |
| BNCI2015_003 | P300 | 10 | run 1 → run 2 |
| MAMEM3 | 5-class SSVEP | 10 | first 5 runs → last 5 runs |

- **Also confirmatory if they download:** PhysionetMI, BI2013a, Wang2016 and EPFLP300.
  Each timed out or was refused by its server during the loading check. A dataset that
  still fails after 3 attempts is excluded for that reason.
- **Excluded:**
  - Shin2017A: about 30 GB, over the disk limit.
  - Cho2017 and Torres: class-blocked storage.
  - Nakanishi2015: a single run leaves classes out of training.
  - Zhou2016: unreadable with the frozen MNE version.

### Code fingerprint at freeze (SHA-256, first 16 hex characters)

| File | Hash |
|---|---|
| `src/processing_eeg_methods/ensemble/__init__.py` | `3096dcb736009476` |
| `src/processing_eeg_methods/ensemble/features.py` | `23143eb14ed69f79` |
| `src/processing_eeg_methods/ensemble/voters.py` | `0097aaf4acd48e47` |
| `src/processing_eeg_methods/ensemble/voting_ensemble.py` | `0020131da453e197` |
| `src/processing_eeg_methods/ensemble/benchmark_moabb.py` | `734be08b3b157381` |
| `src/processing_eeg_methods/ensemble/analyze_voters.py` | `e4227d9a4810d90b` |
| `requirements-ensemble.lock.txt` | `915411c8b4ac775b` |

### Inclusion rules
- Public; loadable with the frozen environment.
- At least 2 sessions, or at least 2 runs in true recording order.
- Total download under ~25 GB for the whole confirmatory set (disk limit).

### Exclusion rules
- **Datasets** whose trials are stored grouped by class (recording order lost) or whose
  chronological split leaves a class out of training.
- **Subjects** whose training or test data lacks a class. These are reported, not hidden.

## Amendment A1: extended voter pool (written 2026-10-26)

**Context.**
- The run of the original 12 voters on the confirmatory datasets ("v1") goes ahead
  unchanged and remains the primary test of H1–H3.
- This amendment adds a second, separately tested configuration ("v2").
- **Disclosure:** when this amendment was written, the v1 confirmatory log had already
  printed the results of 9 subjects of BNCI2015_004. The v2 selection rule below uses
  development data only, so those lines cannot influence it.

**Candidate pool (20 voters):**
- the 12 frozen voters;
- 8 new voters in `extra_voters.py`:
  - xDAWN covariances + tangent space;
  - ERP covariances + MDM;
  - time-delay (augmented) covariances + tangent space;
  - tangent space + SVM;
  - log band power + extra trees;
  - ATCNet, FBCNet and EEGInception-ERP (braindecode, 300 epochs, same training loop).
- The new voters are trained with identical splits and folds (`benchmark_extra.py`).

**Selection of v2 (development datasets only, 52 subjects, deterministic):**
- Criterion: the mean over subjects of the auto-ensemble balanced accuracy, computed from
  the saved probabilities exactly as in `analyze_voters.ensemble_accuracy`.
- Start from all 20 voters and use greedy backward elimination. At each step, remove the
  voter whose removal gives the highest criterion, if it improves the criterion by at
  least 0.1 percentage points; otherwise stop.
- The voters that remain are v2. Implemented in `select_voters.py`.

**Confirmatory tests of v2 (the same 5 confirmatory datasets):**
- **H4.** The v2 auto ensemble beats the best single voter of the v2 pool chosen on
  training out-of-fold accuracy. One-sided Wilcoxon, α = 0.05.
- **H5.** The v2 ensemble differs from the v1 ensemble. Two-sided Wilcoxon, α = 0.05.

**Descriptive additions (all configurations):**
- Per-subject gain: mean ± SD and the full per-subject table.
- Per-voter calibration time and single-trial latency, measured on this machine with no
  other GPU job running.
- A "fast" ensemble made of the classical voters of v2 only, for online use.

**Amendment code fingerprint (SHA-256, first 16 hex characters).** The frozen files
above were re-checked unchanged at this point.

| File | Hash |
|---|---|
| `src/processing_eeg_methods/ensemble/extra_voters.py` | `6467363d0f4e0cf0` |
| `src/processing_eeg_methods/ensemble/benchmark_extra.py` | `5c0ef1d2d4861ff6` |
| `src/processing_eeg_methods/ensemble/select_voters.py` | `b59ef69e49d52755` |

**v2 frozen on 2026-10-27, from development data only, before any v2 voter was run on a
confirmatory dataset.**
- Backward elimination over the 52 development subjects removed, in order: CSP+LDA, band
  power + extra trees, tangent space 4–30 Hz, EEGInception-ERP, MDM 8–30 Hz.
- The criterion went from 75.09% (20 voters) to 76.79%. This is optimistic, because it was
  selected on the same data.
- v2 has 15 voters:
  - ATCNet;
  - augmented covariance;
  - coherence;
  - complexity + boosting;
  - tangent space (6 bands);
  - EEGNet;
  - ERP covariance + MDM;
  - ERP time course;
  - FBCNet;
  - GREEN;
  - MiniRocket;
  - relative spectrum;
  - ShallowNet;
  - tangent space + SVM;
  - xDAWN + tangent space.
- Stored in `Results/ensemble_benchmark/v2_voters.json` (SHA-256 `8c09852cbbe8c3c1`).

**Exploratory addition, written 2026-10-29 before any v1 or v2 analysis of the
confirmatory datasets.** `voter_cost_benefit.py` removes each v2 voter in turn and records the
change in the auto ensemble's accuracy, next to the voter's calibration time and latency. It is
descriptive and changes neither H1–H5 nor any frozen file. Motivation: on development data
alone, ATCNet was the most expensive voter and added little (−0.2 ± 0.9 pp when removed). The
v2 run was nevertheless completed on all confirmatory subjects, so that this question is
answered on complete data. While deciding this, ATCNet's single-voter test accuracy on the
confirmatory subjects finished so far was looked at; no ensemble result was.

## Deviations
- **2026-10-26, before any confirmatory result: BI2013a excluded for size.**
  Schirrmeister2017 took 12 GB on disk, not the ~6.5 GB estimated. Together with Weibo2014
  (4 GB), the core set was already near the ~25 GB limit in the inclusion rules.
  BI2013a ships multi-GB session archives, so it was stopped mid-download and its partial
  files deleted. No BI2013a data was loaded or analysed. The remaining optional datasets
  (EPFLP300, PhysionetMI, Wang2016) are downloaded in that order and stop if free disk
  space falls below 6 GB.
- **2026-10-26, applying the rule above (not a deviation):** EPFLP300 (connection reset by
  the server), PhysionetMI (physionet.org timed out) and Wang2016 (zenodo.org timed out)
  each failed 3 download attempts and are excluded. The confirmatory set is therefore the
  five core datasets.
- **2026-10-27: diverged voter abstains (code change to `voting_ensemble.py`).**
  - The v1 confirmatory run stopped at BNCI2015_003 subject 4 with "Input X contains NaN".
  - The raw data has no NaN. Fitting each voter separately showed that GREEN diverged
    on this subject (all test probabilities NaN), which crashed the stacking combiner.
  - Fix: `_proba_full` now returns uniform probabilities when a voter's output is not
    finite, so a diverged network abstains.
  - A scan of all 312 saved probability files found no non-finite value, so the fix
    cannot change any result obtained before it.
  - New hash of `voting_ensemble.py`: `1f5d1e5881c7f62f` (was `0020131da453e197`).
  - BNCI2015_003 (from subject 4) and MAMEM3 (from subject 3) are resumed with the fixed
    code. MAMEM3 had stopped on a figshare download time-out, not on an analysis error.
  - The run that was still in progress (Weibo2014, Schirrmeister2017) had loaded the old
    code. Any dataset it fails on for the same reason is resumed the same way.
- **2026-10-27: Schirrmeister2017 ran out of RAM (no code change).**
  - Subject 2 hit a MemoryError: a 1.2 GB feature array for 128 channels, on a 16 GB
    machine that was running three jobs.
  - Subject 1 had completed and is kept.
  - The dataset is resumed from subject 2 with the same code, with no other v1 job running.
- **2026-10-27: v2 voters ran out of GPU memory on Schirrmeister2017 (no code change).**
  - ATCNet on subject 3 hit "CUDA error: out of memory" while the v1 run was using the
    same 8 GB GPU.
  - The files already saved (subjects 1–2, and the 4 classical voters of subject 3) are
    kept.
  - The run is resumed with the same code; it skips saved files and retries up to 6 times.
  - All other confirmatory datasets completed for the v2 voters.

## Public release (2026-10-30)
Written after all v1, v2 and timing runs were started. Nothing in the method, the data or
the analysis changes.

**Every file the results depend on is unchanged.** It was re-checked against its last
recorded hash: the freeze table, Amendment A1, and the 2026-10-27 deviation for
`voting_ensemble.py`.

**Line-ending-independent hashes.**
- The hashes above were computed on the Windows working copy, which has CRLF line endings
  in some files. A clone on another system has LF line endings, so those raw hashes differ
  there.
- The table below adds a hash computed after converting CRLF to LF. That hash is the same
  on every system.
- `python -m processing_eeg_methods.ensemble.reproduce check` verifies the
  line-ending-independent column.

### Fingerprints at public release

| File | Raw hash (Windows working copy) | Line-ending-independent hash |
|---|---|---|
| `src/processing_eeg_methods/ensemble/__init__.py` | `3096dcb736009476` | `3096dcb736009476` |
| `src/processing_eeg_methods/ensemble/features.py` | `23143eb14ed69f79` | `23143eb14ed69f79` |
| `src/processing_eeg_methods/ensemble/voters.py` | `0097aaf4acd48e47` | `8834ad633d8344d0` |
| `src/processing_eeg_methods/ensemble/voting_ensemble.py` | `1f5d1e5881c7f62f` | `b9ea96c332f2f3b9` |
| `src/processing_eeg_methods/ensemble/benchmark_moabb.py` | `734be08b3b157381` | `3ebef44e54e810be` |
| `src/processing_eeg_methods/ensemble/analyze_voters.py` | `e4227d9a4810d90b` | `e4227d9a4810d90b` |
| `src/processing_eeg_methods/ensemble/extra_voters.py` | `6467363d0f4e0cf0` | `6467363d0f4e0cf0` |
| `src/processing_eeg_methods/ensemble/benchmark_extra.py` | `5c0ef1d2d4861ff6` | `5c0ef1d2d4861ff6` |
| `src/processing_eeg_methods/ensemble/select_voters.py` | `b59ef69e49d52755` | `b59ef69e49d52755` |
| `src/processing_eeg_methods/ensemble/voter_cost_benefit.py` | `63472254fdb22eea` | `63472254fdb22eea` |
| `requirements-ensemble.lock.txt` | `915411c8b4ac775b` | `07f89148521df11f` |

**Added for the public release.** None of these is used by the analyses above.
- `voter_sets.py`: the v1, v2 and fast voter lists, by name.
- `run_own_data.py`: the same split, folds, training and scoring, applied to a user's own
  epoched data.
- `reproduce.py`: runs the steps above in the order they were run.
- `tools/install_green.py`: fetches GREEN at the commit used here and rewrites only its
  imports.

**Removed from the repository.** Code that is not part of this protocol was removed: the
earlier ProcessingMethods system and the imagined-speech leakage audit. It remains in the
git history.

**Exploratory addition, written 2026-10-30, after every confirmatory result was known.**
`explore_selection.py` asks whether one voter, chosen in advance, could replace the ensemble.
It compares the ensemble with three ways of choosing one voter: picked on the subject's own
training data, best on the development datasets overall, and best on the development datasets
of the same paradigm. It also relates the gain to how far the picked voter is above chance. It
reads only the saved v1 results and changes no hypothesis or frozen file. Its results are
exploratory: the questions were asked after the data had been seen.

## Amendment A2: network tuning (written 2026-10-30, before any tuning result existed)

**Context and disclosure.**
- Written after every v1, v2 and exploratory result above was known. The question it answers
  is one a reader will ask of those results: the networks used generic settings (128 Hz,
  1–45 Hz, 300 epochs), so does the ensemble still win when they are tuned?
- Before writing it, a one-day pilot on BNCI2014_001 (a development dataset) trained the
  networks with published-style settings. No confirmatory data was used.
- Nothing above changes. H1–H5 and their results stand as reported.

**Networks:** EEGNet, ShallowNet, GREEN, EEGInception-ERP, FBCNet and ATCNet. Classical
voters are not changed.

**Configurations (2 inputs x 2 training recipes):**

| Name | Input | Training |
|---|---|---|
| paper | 128 Hz, 1–45 Hz (P300: 1–24 Hz) | AdamW, weight decay 1e-2, label smoothing 0.1, batch 32, 300 epochs, cosine |
| paper_input_published_training | as paper | Adam, no weight decay, no smoothing, batch 64, 1000 epochs, cosine |
| native_input_paper_training | recording rate up to 250 Hz, 0.5 Hz to min(100, 0.45 x rate) | as paper |
| native_input_published_training | as native | as published |

- Learning rate 1e-3 (GREEN 3e-3), as in the benchmark. One seed (42). Same trial windows.
- The "paper" configuration is not retrained; its saved probabilities are reused.

**Tuning:** the 5 development datasets only, with the benchmark's chronological split,
subjects and grouped 5-fold out-of-fold estimate.

**Selection rule:** for each paradigm (MI, P300, SSVEP) and network, the configuration with
the highest out-of-fold balanced accuracy on the development training sessions, averaged
over subjects within a dataset and then over datasets. Ties go to the earlier (cheaper) row.
Test sessions are not used.

**Confirmatory run:** the selected configuration of each network, trained once on the 5
confirmatory datasets (the MI rule for BNCI2015_004, Schirrmeister2017 and Weibo2014).

**Planned comparisons** (confirmatory datasets, per subject, Wilcoxon signed-rank, rule auto):
1. The main ensemble with tuned networks vs the voter picked on training data from the same
   pool.
2. The main ensemble with tuned networks vs each tuned network used alone.
3. The main ensemble with tuned networks vs the main ensemble with the benchmark's networks.

"Tuned" replaces the main ensemble's EEGNet, ShallowNet and GREEN with their selected
configuration.

**Code as run (SHA-256, first 16 hex characters, line-ending-independent).** The run used
scripts kept outside the repository. `tune_networks.py` is the same code, with the output
folder moved to `Results/ensemble_benchmark/tuning/` and the two scripts merged. Re-run on
the saved probabilities, it reproduced `selected.json`, the development scores and every
per-subject result exactly.

| File | Hash | Written |
|---|---|---|
| `tune.py` (tune, select, confirm) | `afe6b2ad069d3829` | 2026-10-30, with this amendment |
| `analyze_tuned.py` (planned comparisons) | `9d4628b14196a55c` | 2026-11-03, during the confirmatory run; tested on development data only |
| `selected.json` | `435ec5cde81edf01` | 2026-11-01, from development data only |
| `src/processing_eeg_methods/ensemble/tune_networks.py` | `eaa2f202f23565c6` | 2026-11-14, public version |

**Selected configurations (2026-11-01):**

| Network | MI | P300 | SSVEP |
|---|---|---|---|
| EEGNet | paper input, published training | native input, paper training | paper input, published training |
| ShallowNet | paper input, published training | native input, published training | paper input, published training |
| GREEN | native input, paper training | paper input, published training | native input, published training |
| EEGInception-ERP | native input, published training | native input, published training | paper input, published training |
| FBCNet | native input, published training | paper | paper input, published training |
| ATCNet | native input, published training | paper input, published training | native input, published training |

**Deviations (none changes code, configuration or rule).**
- 2026-10-31: the first tuning attempt stopped on a Zenodo download time-out (Kalunga2016).
  All datasets were downloaded with retries, and the run restarted from scratch.
- 2026-11-06: the machine restarted for an operating-system update during the confirmatory
  run (Schirrmeister2017, subject 6). Finished files were kept and the run resumed on
  2026-11-07. The resumed `select` step reproduced `selected.json` exactly.
- Descriptive addition (2026-11-14, after the results): `tune_networks.analyze` also reports
  each tuned network against its benchmark configuration. EEGInception-ERP has no such row on
  the confirmatory data, because v2 had dropped it and its benchmark configuration was never
  run there.
- Tie handling (2026-11-14, after the results): per-subject differences are rounded to 1e-10
  before the Wilcoxon test, so that equal accuracies tie exactly. Floating-point noise of about
  1e-16 had split such ties. This moved p for comparison 1 from 0.035 to 0.036 and for comparison
  3 from 0.41 to 0.43. The same rounding is now used in `explore_selection.py` and the paper's
  tables. For H1, H4, H5 and the fast ensemble it changes only the third significant digit
  (for example H1: 0.00083 to 0.00080); the frozen scripts that print them are unchanged.
- Ties (2026-11-14): counts of subjects "better" and of subjects whose picked voter was the
  best now treat equal accuracies as ties. The picked voter had the best test accuracy, alone or
  tied, for 23/53 confirmatory subjects; the earlier count of 21 matched voters by name.

**Result (2026-11-14, 53 confirmatory subjects):**
1. The tuned ensemble 63.8% vs the picked voter 62.4%: +1.4 pp, better for 33/53, p = 0.036.
2. It beat every tuned network used alone by 4.7 to 10.4 pp (p ≤ 0.004).
3. Tuned vs benchmark networks in the ensemble: 63.8% vs 63.9%, −0.1 pp, p = 0.43.

## Amendment A3: paradigm-standard voters (written 2026-11-14 15:00, before any of them was run on any dataset)

**Context and disclosure.**
- Written after every result above was known, in answer to a review of the draft: the v1 pool
  has no standard pipeline for SSVEP (CCA, TRCA) and none for P300 in v1 (xDAWN with LDA), so
  its baseline was the best voter of our pool rather than the paradigm's standard method.
- Nothing above changes. H1-H5 and Amendment A2 stand as reported.

**Voters** (`standard_voters.py`; settings from the original papers, no tuning):
- **MI** (motor imagery, mental imagery and motor execution): `fbcsp`. Log-variance of 4 CSP
  components (Ledoit-Wolf) in each 4-Hz band from 4 to 40 Hz, the 8 features with the highest
  mutual information with the class, shrinkage LDA (Ang et al. 2008).
- **P300:** `xdawn_lda`. xDAWN spatial filters (4 per class), the filtered time course
  averaged in bins of 4 samples, shrinkage LDA with equal priors (Rivet et al. 2009; Blankertz
  et al. 2011).
- **SSVEP:** `cca`. Largest canonical correlation with sine/cosine references at each class's
  stimulus frequency and its 2nd harmonic (Lin et al. 2006), followed by logistic regression
  (C = 0.1, class-balanced), so that classes without a frequency (Kalunga2016 "rest") can be
  learned. `trca`: ensemble TRCA (Nakanishi et al. 2018), correlation with each class
  template, then the same logistic regression.
- Each standard voter is added only to the datasets of its paradigm. Same trials, chronological
  split, grouped 5-fold OOF and refit as the benchmark (`standard_voters run`).

**Hypothesis and tests (confirmatory datasets, 53 subjects, rule auto).** "Full pool" = the 12
v1 voters plus the standard voters of the dataset's paradigm.
- **H6.** The full-pool ensemble beats the best single voter of the full pool chosen on
  training OOF accuracy. One-sided Wilcoxon signed-rank on per-subject differences, alpha = 0.05.
- **Secondary** (two-sided Wilcoxon, Holm-adjusted together): the full-pool ensemble vs the v1
  ensemble (53 subjects); the full-pool ensemble vs each standard voter alone, on the subjects
  of its paradigm (FBCSP: 33; xDAWN + LDA: 10; CCA and TRCA: 10).
- The development datasets are reported descriptively.
- Per-subject differences are rounded to 1e-10 before testing, so that equal accuracies tie.

**Code fingerprint (SHA-256, first 16 hex characters, line-ending-independent).**

| File | Hash |
|---|---|
| `src/processing_eeg_methods/ensemble/standard_voters.py` | `b1dd12c90a2e9e78` |

**Run.** 2026-11-14 15:02 to 17:45, `standard_voters run` on all ten datasets, CPU only. The
fingerprint above was re-checked after the run and is unchanged. No deviation from this
amendment.

**Result (2026-11-14, 53 confirmatory subjects):**
- **H6 supported.** Full-pool ensemble 67.9% vs the full-pool picked voter 66.9%:
  +1.1 ± 6.8 pp, 36 of 53 subjects improved, one-sided p = 0.039.
- Secondary (Holm-adjusted): full vs v1 ensemble +4.1 pp (p = 0.030); vs FBCSP alone +14.2 pp
  (p = 2.1e-5); vs TRCA alone +27.0 pp (p = 0.023); vs xDAWN + LDA alone +2.0 pp (p = 0.32);
  vs CCA alone -5.0 pp (p = 0.32).
- On SSVEP (MAMEM3) the ensemble (47.0%) scored below CCA alone (52.1%) and below the picked
  voter (50.7%); the same held on the development SSVEP dataset (Kalunga2016). TRCA was at
  chance level on both.
- Full tables: `Results/ensemble_benchmark/standard/RESULTS.md` (`standard_voters analyze`) and
  docs/SUPPLEMENTARY.md, table S7.
