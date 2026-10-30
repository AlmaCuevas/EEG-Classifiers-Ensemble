"""Feature transformers for epoched EEG, shaped (trials, channels, samples).

Each transformer is a scikit-learn TransformerMixin: every statistic (re-centering
reference, z-score means/stds, CSP-like filters) is learned in fit() from the training
trials only and re-used in transform(), so they are safe inside cross-validation.
They look at different aspects of the signal so that voters built on them make
different mistakes:
    FilterBankTangentSpace : spatial covariance per band (Riemannian geometry)
    ERPFeatures            : low-frequency time course (evoked responses)
    RelativeSpectrum       : fine-resolution relative power spectrum per channel
    ComplexityFeatures     : Hjorth parameters and entropies (signal complexity)
    CoherenceFeatures      : functional connectivity between channel pairs
"""

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from scipy.signal import butter, coherence, sosfiltfilt, welch
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))


def bandpass(x, l_freq, h_freq, sfreq):
    h_freq = min(h_freq, 0.45 * sfreq)
    sos = butter(4, [l_freq, h_freq], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


class _ZScore:
    """z-score helper whose statistics are learned on training features only."""

    def fit(self, f):
        self.mean_, self.std_ = f.mean(0), f.std(0) + 1e-9
        return self

    def transform(self, f):
        return (f - self.mean_) / self.std_


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Band-wise covariance matrices, re-centered to identity with the training mean
    (Riemannian alignment), projected to the tangent space."""

    def __init__(self, sfreq=250.0, bands=DEFAULT_BANDS, estimator="oas"):
        self.sfreq, self.bands, self.estimator = sfreq, bands, estimator

    def _covs(self, x):
        return [Covariances(self.estimator).transform(bandpass(x, lo, hi, self.sfreq))
                for lo, hi in self.bands]

    def fit(self, X, y=None):
        self.whiteners_ = [invsqrtm(mean_riemann(c)) for c in self._covs(X)]
        return self

    def transform(self, X):
        out = []
        for c, w in zip(self._covs(X), self.whiteners_):
            aligned = w @ c @ w
            out.append(tangent_space(aligned, np.eye(aligned.shape[-1])))
        return np.concatenate(out, axis=1)


class ERPFeatures(BaseEstimator, TransformerMixin):
    """Low-pass time course of every channel (per-trial demeaned), decimated."""

    def __init__(self, sfreq=250.0, l_freq=1.0, h_freq=12.0, n_points=32):
        self.sfreq, self.l_freq, self.h_freq, self.n_points = sfreq, l_freq, h_freq, n_points

    def _raw(self, X):
        xf = bandpass(X, self.l_freq, self.h_freq, self.sfreq)
        xf = xf - xf.mean(axis=-1, keepdims=True)
        idx = np.linspace(0, xf.shape[-1] - 1, self.n_points).astype(int)
        return xf[:, :, idx].reshape(len(X), -1)

    def fit(self, X, y=None):
        self.z_ = _ZScore().fit(self._raw(X))
        return self

    def transform(self, X):
        return self.z_.transform(self._raw(X))


class RelativeSpectrum(BaseEstimator, TransformerMixin):
    """Log relative power in 1 Hz-ish bins between f_min and f_max, per channel."""

    def __init__(self, sfreq=250.0, f_min=2.0, f_max=40.0):
        self.sfreq, self.f_min, self.f_max = sfreq, f_min, f_max

    def _raw(self, X):
        nperseg = min(int(self.sfreq // 2), X.shape[-1])
        freqs, p = welch(bandpass(X, 1, 45, self.sfreq), fs=self.sfreq, nperseg=nperseg,
                         noverlap=nperseg * 3 // 4, axis=-1)
        keep = (freqs >= self.f_min) & (freqs <= self.f_max)
        p = p[:, :, keep]
        return np.log(p / p.sum(axis=-1, keepdims=True)).reshape(len(X), -1)

    def fit(self, X, y=None):
        self.z_ = _ZScore().fit(self._raw(X))
        return self

    def transform(self, X):
        return self.z_.transform(self._raw(X))


class ComplexityFeatures(BaseEstimator, TransformerMixin):
    """Hjorth mobility/complexity and log-variance per band, plus permutation and
    spectral entropy (requires the optional package ``antropy``)."""

    def __init__(self, sfreq=250.0, bands=((1, 40), (4, 8), (8, 13), (13, 30))):
        self.sfreq, self.bands = sfreq, bands

    def _raw(self, X):
        import antropy

        feats = []
        for lo, hi in self.bands:
            xf = bandpass(X, lo, hi, self.sfreq)
            mobility, complexity = antropy.hjorth_params(xf, axis=-1)
            feats += [mobility, complexity, np.log(xf.var(axis=-1))]
        xf = bandpass(X, 1, 40, self.sfreq)
        feats.append(np.array([[antropy.perm_entropy(c, normalize=True) for c in t] for t in xf]))
        feats.append(np.array([[antropy.spectral_entropy(c, sf=self.sfreq, method="welch", normalize=True)
                                for c in t] for t in xf]))
        return np.concatenate(feats, axis=1)

    def fit(self, X, y=None):
        self.z_ = _ZScore().fit(self._raw(X))
        return self

    def transform(self, X):
        return self.z_.transform(self._raw(X))


class CoherenceFeatures(BaseEstimator, TransformerMixin):
    """Magnitude-squared coherence of every channel pair, averaged per band."""

    def __init__(self, sfreq=250.0, bands=((4, 8), (8, 13), (13, 30))):
        self.sfreq, self.bands = sfreq, bands

    def _raw(self, X):
        xf = bandpass(X, 1, 40, self.sfreq)
        iu = np.triu_indices(X.shape[1], 1)
        nperseg = min(int(self.sfreq // 2), X.shape[-1])
        out = []
        for t in xf:
            freqs, c = coherence(t[iu[0]], t[iu[1]], fs=self.sfreq, nperseg=nperseg, axis=-1)
            out.append(np.concatenate([c[:, (freqs >= lo) & (freqs < hi)].mean(1) for lo, hi in self.bands]))
        return np.array(out)

    def fit(self, X, y=None):
        self.z_ = _ZScore().fit(self._raw(X))
        return self

    def transform(self, X):
        return self.z_.transform(self._raw(X))
