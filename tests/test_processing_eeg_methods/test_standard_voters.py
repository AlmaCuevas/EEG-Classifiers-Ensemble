import numpy as np
import pytest

from processing_eeg_methods.ensemble.standard_voters import (
    STANDARD,
    CCAFeatures,
    TRCAFeatures,
    frequencies,
    standard_voters,
)

SFREQ = 128.0


def ssvep(freqs=(8.0, 10.0, 12.0), n_per_class=12, n_channels=4, seconds=2.0, seed=0):
    """Trials of noise plus a sine at the class frequency on every channel (random phase)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    X, y = [], []
    for k, f in enumerate(freqs):
        for _ in range(n_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            X.append(
                rng.standard_normal((n_channels, len(t)))
                + np.sin(2 * np.pi * f * t + phase)
            )
            y.append(k)
    return np.array(X), np.array(y)


def test_frequencies_skip_classes_without_one():
    assert frequencies(["13", "17", "21", "rest"]) == (13.0, 17.0, 21.0)
    assert frequencies(["6.66", "7.50"]) == (6.66, 7.5)


def test_cca_peaks_at_the_stimulus_frequency():
    X, y = ssvep()
    features = CCAFeatures(SFREQ, (8.0, 10.0, 12.0)).transform(X)
    assert (features.argmax(1) == y).mean() > 0.9


def test_trca_separates_class_locked_responses():
    rng = np.random.default_rng(1)
    templates = rng.standard_normal((3, 4, 128))
    y = np.repeat(np.arange(3), 15)
    X = templates[y] + 2.0 * rng.standard_normal((len(y), 4, 128))
    features = TRCAFeatures().fit(X[::2], y[::2]).transform(X[1::2])
    assert (features.argmax(1) == y[1::2]).mean() > 0.9


@pytest.mark.parametrize("paradigm", list(STANDARD))
def test_every_standard_voter_returns_probabilities(paradigm):
    if paradigm == "SSVEP":
        X, y = ssvep()
        names = ["8", "10", "12"]
    else:
        rng = np.random.default_rng(2)
        y = np.repeat([0, 1], 40)
        X = rng.standard_normal((80, 8, 128))
        X[y == 1, :2] *= 2.0
        names = ["a", "b"]
    for name, voter in standard_voters(paradigm, SFREQ, names).items():
        proba = voter.fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), len(names)), name
        assert np.allclose(proba.sum(1), 1.0), name
