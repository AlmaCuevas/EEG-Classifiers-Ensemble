import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from processing_eeg_methods.ensemble import VotingEnsemble, default_voters

warnings.filterwarnings("ignore")
N_CLASSES, N_CHANNELS, N_SAMPLES, SFREQ = 4, 6, 200, 100.0


def channel_power(channel):
    def f(x):
        return np.log(x[:, channel, :].var(axis=1, keepdims=True) + 1e-9) * [1, 1]

    return f


def make_independent_views(n_trials=600, seed=0, gain=0.35):
    """Each channel carries a weak, independent hint about the class: channel c has
    more power when the trial's class equals c % 4. Voters reading different channels
    therefore make (mostly) independent errors."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n_trials)
    x = rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))
    for c in range(N_CHANNELS):
        x[y == c % N_CLASSES, c] *= 1 + gain
    return x, y


def channel_voters():
    return {
        f"ch{c}": make_pipeline(
            FunctionTransformer(channel_power(c)),
            StandardScaler(),
            LogisticRegression(),
        )
        for c in range(N_CHANNELS)
    }


def test_probabilities_are_valid():
    x, y = make_independent_views(200)
    ens = VotingEnsemble(channel_voters()).fit(
        x[:150], y[:150], groups=np.arange(150) // 10
    )
    p = ens.predict_proba(x[150:])
    assert p.shape == (50, N_CLASSES)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert (p >= 0).all()


@pytest.mark.parametrize("combination", ["weighted", "soft", "majority", "stacking"])
def test_ensemble_beats_average_voter_when_errors_are_independent(combination):
    x, y = make_independent_views(1200)
    tr, te = slice(0, 800), slice(800, None)
    ens = VotingEnsemble(channel_voters(), combination=combination).fit(
        x[tr], y[tr], groups=np.arange(800) // 20
    )
    ens_acc = balanced_accuracy_score(y[te], ens.predict(x[te]))
    voter_accs = [
        balanced_accuracy_score(y[te], ens.classes_[p.argmax(1)])
        for p in ens.voter_proba(x[te]).values()
    ]
    assert ens_acc > np.mean(voter_accs) + 0.05


def test_grouped_weights_ignore_a_drift_cheater():
    """Trials come in blocks of 4 with the same label and a shared block offset (like
    consecutive repetitions in imagined-speech recordings). A 1-NN on the offset
    'decodes' the label under shuffled CV but not when blocks are kept together."""
    rng = np.random.default_rng(1)
    n_blocks = 100
    y = np.repeat(rng.integers(0, N_CLASSES, n_blocks), 4)
    blocks = np.repeat(np.arange(n_blocks), 4)
    x = rng.standard_normal((len(y), N_CHANNELS, N_SAMPLES))
    x += (
        rng.standard_normal((n_blocks, N_CHANNELS, 1))[blocks] * 5
    )  # block offset, no class info
    cheater = {
        "offset_1nn": make_pipeline(
            FunctionTransformer(lambda a: a.mean(2)), KNeighborsClassifier(1)
        )
    }
    shuffled = VotingEnsemble(cheater).fit(x, y)  # warns: no groups
    grouped = VotingEnsemble(cheater).fit(x, y, groups=blocks)
    assert shuffled.oof_accuracy_["offset_1nn"] > 0.6
    assert grouped.oof_accuracy_["offset_1nn"] < 0.35


def test_accumulation_over_repetitions_improves_decisions():
    x, y = make_independent_views(1600, gain=0.04)
    ens = VotingEnsemble(channel_voters()).fit(
        x[:800], y[:800], groups=np.arange(800) // 20
    )
    xt, yt = x[800:], y[800:]
    rng = np.random.default_rng(0)
    single, accumulated = [], []
    for _ in range(200):
        c = rng.integers(N_CLASSES)
        pick = rng.choice(np.flatnonzero(yt == c), 8, replace=False)
        single.append(ens.predict(xt[pick[:1]])[0] == c)
        accumulated.append(ens.predict_accumulated(xt[pick]) == c)
    assert np.mean(accumulated) > np.mean(single) + 0.1


def test_diversity_report_columns():
    x, y = make_independent_views(200)
    ens = VotingEnsemble(channel_voters()).fit(x, y, groups=np.arange(200) // 10)
    report = ens.diversity_report()
    assert list(report.columns) == [
        "oof_balanced_accuracy",
        "weight",
        "mean_error_correlation_with_others",
    ]
    assert np.isclose(report["weight"].sum(), 1)


def test_default_voters_run_end_to_end():
    x, y = make_independent_views(120, gain=1.0)
    voters = default_voters(sfreq=SFREQ, include_optional=False)
    ens = VotingEnsemble(voters, n_splits=3).fit(x, y, groups=np.arange(120) // 10)
    assert ens.predict_proba(x[:5]).shape == (5, N_CLASSES)
    assert set(ens.diversity_report().index) == set(voters)
