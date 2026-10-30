"""Leakage-safe voting ensemble of diverse EEG classifiers.

Why voting can help, and when it cannot (Condorcet's jury theorem): combining voters
improves accuracy only if each voter is better than chance AND their mistakes are not
strongly correlated. VotingEnsemble therefore
  * estimates every voter's accuracy OUT-OF-FOLD (never on its own training trials),
    using a group-aware split so trials from the same block/run never sit on both sides;
  * weights votes by out-of-fold accuracy above chance (voters at chance get no vote);
  * can learn a stacking combiner on the out-of-fold probabilities;
  * reports the error correlation between voters (diversity_report) so you can see
    whether voting can help at all;
  * can accumulate evidence over repeated trials of the same intended class (online use).

Typical use:
    ens = VotingEnsemble(default_voters(sfreq=250))  # combination="auto"
    ens.fit(X_train, y_train, groups=runs_train)     # groups = runs/blocks/sessions
    proba = ens.predict_proba(X_new)
    print(ens.diversity_report())
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

COMBINATIONS = ("auto", "weighted", "soft", "majority", "stacking")


def _proba_full(model, X, n_classes):
    """predict_proba aligned to the encoded classes 0..n_classes-1."""
    p = model.predict_proba(X)
    if not np.isfinite(p).all():  # a diverged network abstains (uniform vote)
        return np.full((len(X), n_classes), 1.0 / n_classes)
    out = np.full((len(X), n_classes), 1e-9)
    out[:, np.asarray(model.classes_, dtype=int)] = p
    return out / out.sum(1, keepdims=True)


def _oof_one(voter, X, y, splits, n_classes):
    warnings.filterwarnings("ignore")
    oof = np.zeros((len(y), n_classes))
    for tr, te in splits:
        oof[te] = _proba_full(clone(voter).fit(X[tr], y[tr]), X[te], n_classes)
    return oof


def _fit_one(voter, X, y):
    warnings.filterwarnings("ignore")
    return clone(voter).fit(X, y)


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """Parameters
    ----------
    voters : dict name -> classifier (see voters.default_voters)
    combination : "auto" (default), "weighted", "soft", "majority" or "stacking".
        "auto" = stacking when there are at least min_trials_per_class_for_stacking
        training trials per class, otherwise weighted. Across 5 public datasets
        (motor imagery, P300, SSVEP; cross-session) stacking gave the largest gains
        with enough data and overfitted with very little (see README).
    n_splits : folds of the internal out-of-fold estimate
    n_jobs : parallel voters
    min_trials_per_class_for_stacking : below this, "stacking" falls back to "weighted"
    """

    def __init__(self, voters=None, combination="auto", n_splits=5, n_jobs=1,
                 min_trials_per_class_for_stacking=40):
        self.voters, self.combination, self.n_splits, self.n_jobs = voters, combination, n_splits, n_jobs
        self.min_trials_per_class_for_stacking = min_trials_per_class_for_stacking

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, groups=None):
        if self.combination not in COMBINATIONS:
            raise ValueError(f"combination must be one of {COMBINATIONS}")
        if groups is None:
            warnings.warn(
                "VotingEnsemble.fit without `groups`: out-of-fold weights use a shuffled split, "
                "which over-estimates accuracy when neighbouring trials are correlated. Pass the "
                "run/block/session of every trial as `groups`.", stacklevel=2)
        X = np.asarray(X)
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        yy = self.encoder_.transform(y)
        n_classes = len(self.classes_)
        self.chance_ = 1.0 / n_classes
        if groups is not None:
            splits = list(GroupKFold(min(self.n_splits, len(np.unique(groups)))).split(X, yy, groups))
        else:
            splits = list(StratifiedKFold(self.n_splits, shuffle=True, random_state=42).split(X, yy))

        names = list(self.voters)
        oofs = Parallel(n_jobs=self.n_jobs)(
            delayed(_oof_one)(self.voters[n], X, yy, splits, n_classes) for n in names)
        self.oof_proba_ = dict(zip(names, oofs))
        self.oof_accuracy_ = {n: balanced_accuracy_score(yy, p.argmax(1)) for n, p in self.oof_proba_.items()}
        raw = np.array([max(self.oof_accuracy_[n] - self.chance_, 0.0) for n in names])
        self.weights_ = dict(zip(names, raw / raw.sum() if raw.sum() > 0 else np.full(len(names), 1 / len(names))))
        self.combination_used_ = "stacking" if self.combination == "auto" else self.combination
        if self.combination_used_ == "stacking":
            if np.bincount(yy).min() < self.min_trials_per_class_for_stacking:
                # A learned combiner over many voters overfits small datasets (e.g. 8 trials
                # per class collapsed SSVEP accuracy below chance); fall back to weighting.
                if self.combination == "stacking":
                    warnings.warn("Too few trials per class for stacking; using weighted voting.", stacklevel=2)
                self.combination_used_ = "weighted"
            else:
                self.meta_ = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(
                    self._meta_features(self.oof_proba_), yy)
        self._oof_y = yy
        self.fitted_voters_ = dict(zip(names, Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(self.voters[n], X, yy) for n in names)))
        return self

    # -------------------------------------------------------------- predict
    def _meta_features(self, probas):
        return np.log(np.concatenate([probas[n] for n in self.voters], axis=1) + 1e-9)

    def voter_proba(self, X):
        n_classes = len(self.classes_)
        return {n: _proba_full(m, np.asarray(X), n_classes) for n, m in self.fitted_voters_.items()}

    def _combine(self, probas, combination=None):
        combination = combination or getattr(self, "combination_used_", self.combination)
        names = list(probas)
        stack = np.stack([probas[n] for n in names])  # (voters, trials, classes)
        if combination == "soft":
            return stack.mean(0)
        if combination == "weighted":
            w = np.array([self.weights_[n] for n in names])
            return np.tensordot(w, stack, 1)
        if combination == "majority":
            votes = stack.argmax(2)
            counts = np.stack([(votes == c).sum(0) for c in range(stack.shape[2])], axis=1).astype(float)
            counts += 1e-3 * stack.mean(0)  # break ties with the mean probability
            return counts / counts.sum(1, keepdims=True)
        p = self.meta_.predict_proba(self._meta_features(probas))
        return p

    def predict_proba(self, X):
        return self._combine(self.voter_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_accumulated(self, X_repetitions):
        """One decision from several trials of the SAME intended class (e.g. the user
        thinks the word N times). Combines evidence by summing log-probabilities."""
        log_p = np.log(self.predict_proba(X_repetitions) + 1e-9).sum(0)
        return self.classes_[int(np.argmax(log_p))]

    # ------------------------------------------------------------ reporting
    def diversity_report(self) -> pd.DataFrame:
        """Out-of-fold accuracy, vote weight and mean error correlation of every voter.
        Voting can only help when accuracies are above chance and correlations are low."""
        names = list(self.oof_proba_)
        correct = np.array([(self.oof_proba_[n].argmax(1) == self._oof_y) for n in names], dtype=float)
        corr = np.corrcoef(correct)
        np.fill_diagonal(corr, np.nan)
        return pd.DataFrame({
            "oof_balanced_accuracy": [self.oof_accuracy_[n] for n in names],
            "weight": [self.weights_[n] for n in names],
            "mean_error_correlation_with_others": np.nanmean(corr, axis=1),
        }, index=names).sort_values("oof_balanced_accuracy", ascending=False)
