# What we learned about voting ensembles for EEG

This document collects the findings from the benchmark behind this repository, including
practical lessons that do not fit in a paper.

All numbers come from:
- 10 public MOABB datasets: motor imagery and execution, mental imagery, P300 and SSVEP;
- one model per subject, trained on the first session and tested on later sessions or runs;
- balanced accuracy.

**Confirmatory** results come from 5 datasets (53 subjects) that played no part in designing
the method. **Exploratory** marks analyses chosen after those results were known.

Every number can be regenerated with `python -m processing_eeg_methods.ensemble.reproduce
analyze`. The script that produces each result is named in brackets.

## 1. The ensemble beats the voter you would have picked

| Confirmatory data, 53 subjects | Accuracy |
|---|---|
| Best single voter, picked on each subject's training data | 61.5% |
| Ensemble (`auto`, 12 voters) | **63.9%** |

- **Gain:** +2.4 ± 5.5 points (95% CI +1.0 to +3.8).
- **Subjects:** 36 improved, 15 worse, 2 tied; one-sided Wilcoxon p = 0.0008.
- **Datasets:** every confirmatory dataset improved on average, across all five datasets.
- **Development data:** the gain was +4.1 points; data used to make choices is optimistic.

[`analyze_voters`]

## 2. Why not just use the best voter? You cannot know it in advance

*Exploratory* [`explore_selection`]:
- **No voter is best for most subjects.** On the confirmatory data, 12 different voters were
  the best on the test data for at least one subject. The most frequent, GREEN, was best for
  only 10 of 53.
- **Training data rarely identifies the winner.** The voter that looked best on a subject's
  own training data was the true best for only 21 of 53 subjects (40%).
- **Educated guesses do worse than voting.**

| One voter, chosen as | Accuracy | Ensemble better in | Ensemble minus it |
|---|---|---|---|
| Best on the subject's own training data | 61.5% | 36/53 | +2.4 points (p = 0.002) |
| Best on other datasets of the same paradigm (GREEN for MI, EEGNet for P300, relative spectrum for SSVEP) | 60.0% | 35/53 | +3.8 points (p = 0.013) |
| Best on other datasets overall (tangent space) | 56.4% | 44/53 | +7.4 points (p < 0.001) |
| Best on the test data (hindsight; impossible in practice) | 65.4% | 25/53 | −1.6 points (p = 0.14, not significant) |

*p: two-sided Wilcoxon.*

- **The ensemble's value is hedging an unreliable choice.** The hindsight-best voter was 4.0
  points above the picked one, and the ensemble recovered 60% of that gap without knowing the
  answer. [`analyze_voters`]
- **It costs no extra calibration.** Choosing one voter honestly already requires training
  every candidate and estimating its out-of-fold accuracy, which is all the ensemble needs.
  Voting adds only the prediction time of the other voters.

## 3. Voters must be above chance, not necessarily good

*Exploratory* [`explore_selection`]:
- **The gain did not depend on how far the voters were above chance** (Spearman ρ = −0.04,
  p = 0.70, 105 subjects).

| Picked voter above chance | Subjects | Mean gain | Improved |
|---|---|---|---|
| < 5 points | 9 | +3.4 | 6/9 |
| 5–15 points | 13 | +2.0 | 7/13 |
| 15–30 points | 40 | +3.9 | 28/40 |
| > 30 points | 43 | +3.0 | 36/43 |

- **This is Condorcet's jury theorem.** Combining voters amplifies whatever signal they share,
  so a vote among voters that are each above chance beats its members. With two classes,
  above chance means right more than half the time. With more classes, it means picking the
  right class more often than any single wrong one.
- **There is no fixed accuracy threshold.** The size of the gain depends mainly on how
  independent the voters' mistakes are. Ours had a mean error correlation of 0.17, hence gains
  of a few points rather than dramatic ones.
- **At chance there is nothing to amplify.** On imagined speech, every voter was at chance
  under a leakage-free protocol, and the ensemble gave nothing.
- **The largest single-subject losses** (more than 5 points, 4 subjects) all occurred in the
  two datasets closest to chance, where per-subject estimates are noisiest.
- **The weighting guards against useless voters.** A voter at or below chance on the training
  data gets weight zero, so it cannot drag the vote down.

## 4. Diversity as such did not predict the gain

- **Error correlation did not predict the gain.** How correlated a subject's voters' errors
  were showed no relation to its gain (H2: Spearman ρ = 0.03, p = 0.59).
- **More voters helped up to a point.** With 6 voters the gain was already +2.4 points; with
  all 12 it was the same. [`analyze_voters`]
- **Mixing families helped.** Classical voters alone and network voters alone each reached
  61.8%, while all 12 together reached 63.9%. [`analyze_voters`]

## 5. How to combine the votes matters more than which voters you add

| Rule (confirmatory) | Accuracy |
|---|---|
| Majority vote | 59.7% |
| Soft vote (mean of probabilities) | 59.3% |
| Weighted by out-of-fold accuracy above chance | 63.6% |
| Stacking (logistic regression on out-of-fold probabilities) | 63.6% |
| `auto` (stacking if ≥ 40 training trials per class, else weighted) | **63.9%** |

- **Simple averaging fell below the single picked voter (61.5%).** It gives a weak voter the
  same say as a strong one. Use `weighted`, `stacking` or `auto`.
- **Stacking needs data.** With 8 trials per class (Kalunga2016), stacking collapsed;
  weighted voting did not.

## 6. Do not curate the voter pool on small data

- **The experiment** [`select_voters`]:
  - eight more voters were added, giving 20;
  - 15 were chosen by backward elimination on the development data ("v2").
- **The result:**
  - on the development data v2 looked 2.6 points better than the original 12;
  - on the confirmatory data it was 0.2 points worse (p = 0.71).
- **Voter-level effects did not transfer either.** CSP+LDA looked harmful on the development
  data and helpful on the confirmatory data.
- **Keep a broad pool,** and let the out-of-fold weights do the selecting.

## 7. Speed and online use

Measured on one laptop (RTX 4060, 8 GB), subject 1 of each dataset, nothing else running,
voters run one after another [`reproduce timing`]:

| | Prediction per trial, median (max) | Calibration per subject, median (max) |
|---|---|---|
| `fast` (10 classical voters) | 34 ms (655 ms) | 1.5 min (52 min) |
| `v2` (15 voters incl. 5 networks) | 52 ms (679 ms) | 11.3 min (98 min) |

- **Prediction is fast.** The median voter classifies a trial in a few milliseconds. Even the
  whole ensemble run serially needs at most 17% of a trial's length, so it fits in an online
  loop. The maxima come from the 128-channel Schirrmeister2017 data, mostly from the coherence
  voter (0.43 s).
- **Calibration is the bottleneck,** and the neural networks dominate it. ATCNet had the
  largest median calibration time (3.5 min); GREEN the largest single one (29 min on
  BNCI2015_003, which has 900 training trials).
- **The `fast` set** (the 10 classical voters of v2, no GPU) scored 62.1% on the confirmatory
  data:
  - +1.7 ± 4.7 points over its own best voter, improving 35/53 subjects (p = 0.002);
  - 1.6 points below the full ensemble.

  It is the sensible default for systems that recalibrate every session. [`select_voters`]
- **ATCNet was the most expensive voter.** Removing it changed the ensemble by +0.1 ± 2.8
  points on the confirmatory data, so it is not worth its cost. [`voter_cost_benefit`]
- **The voters whose removal hurt most** were augmented covariance (−0.7 points), FBCNet
  (−0.5) and ShallowNet (−0.4). [`voter_cost_benefit`]
- **Per-voter costs** are written to `Results/ensemble_benchmark/timing/timing.csv`.

## 8. How this compares with published results

**Most published numbers are not comparable.** They come from easier protocols:
- shuffled cross-validation within one session (the MOABB benchmark);
- leaving one session out, which trains on later sessions too;
- picking the model or epoch on the test session.

**Two datasets can be compared,** because our split is the standard one there:
BNCI2014_001 (session 1 → session 2) and Schirrmeister2017 (official training runs → test runs).

| Dataset | Ensemble | Published with the same split (accuracy) |
|---|---|---|
| BNCI2014_001 (development) | 76.5% | ATCNet 79.1%, EEG Conformer 75.8%, EEGNet 71.1% (Wimpff et al., J. Neural Eng. 2024); ShallowConvNet 69.6%, validation-locked (Zhou & Roy 2026, preprint) |
| Schirrmeister2017 (confirmatory) | 92.3% | EEG Conformer 95.3%, ATCNet 92.6%, ShallowNet 88.7%, EEGNet 84.7% (Wimpff et al. 2024) |

**What the comparison shows:**
- **The ensemble reaches the level of tuned networks.** It uses generic, untuned voters and
  one preprocessing pipeline for every paradigm (1–45 Hz, 128 Hz).
- **Its own ATCNet voter scored lower** on the same data and settings: 86.5% on
  Schirrmeister2017 and 55.1% on BNCI2014_001.
- **The published networks have advantages here.** They use the original sampling rate,
  dataset-specific training, and (Schirrmeister2017) frequencies up to 125 Hz.
- **Status:** competitive with the state of the art, not above it, and without tuning.

## 9. Practical guidance

- **Pass `groups`** (the run, block or session of each trial). Without it, the out-of-fold
  weights are estimated on shuffled folds and inflated by neighbouring, correlated trials.
- **Keep trials in recording order,** and test on data recorded after the training data.
- **Start with `get_voters("fast")`.** Use `v1` with a GPU when the model is trained once and
  reused.
- **Check the condition before expecting a gain.** If every voter's out-of-fold accuracy is
  at chance, the ensemble cannot help; look at the data, not the model.
- **Prefer `combination="auto"`.**

## 10. Datasets we could not use without leakage

- **Zhou2016:** unreadable with current MNE.
- **Nakanishi2015:** a single run, so a chronological split leaves classes out of training.
- **Cho2017:** trials are stored grouped by class, so the recording order is lost.

The pre-registration, its deviations and the code fingerprints are in
[PROTOCOL.md](src/processing_eeg_methods/ensemble/PROTOCOL.md).
