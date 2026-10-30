"""Candidate voters added after the pre-registration freeze (exploratory, see PROTOCOL.md
amendment A1). Kept out of voters.py so the frozen code and its fingerprint stay unchanged.

Classical: xDAWN covariances, ERP (prototype) covariances, time-delay ("augmented")
covariances, tangent space + SVM, band power + extra trees.
Deep (braindecode): ATCNet, FBCNet, EEGInception-ERP.
"""

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from processing_eeg_methods.ensemble.features import bandpass
from processing_eeg_methods.ensemble.voters import BandCovariances, TorchVoter, _lr


class TimeDelayCovariances(BaseEstimator, TransformerMixin):
    """Covariance of the signal stacked with delayed copies of itself (augmented covariance),
    so the matrix also captures temporal structure. The embedding order shrinks with the
    channel count to keep the matrix small."""

    def __init__(self, sfreq=128.0, l_freq=8.0, h_freq=30.0, lag=4):
        self.sfreq, self.l_freq, self.h_freq, self.lag = sfreq, l_freq, h_freq, lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = bandpass(X, self.l_freq, self.h_freq, self.sfreq)
        order = max(1, min(4, 48 // x.shape[1]))
        n = x.shape[2] - (order - 1) * self.lag
        embedded = np.concatenate([x[:, :, i * self.lag : i * self.lag + n] for i in range(order)], axis=1)
        return Covariances("oas").transform(embedded)


class LogBandPower(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=128.0, bands=((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))):
        self.sfreq, self.bands = sfreq, bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nyquist = 0.45 * self.sfreq
        feats = [np.log(bandpass(X, lo, min(hi, nyquist), self.sfreq).var(-1) + 1e-12)
                 for lo, hi in self.bands if lo < nyquist]
        return np.concatenate(feats, axis=1)


class BraindecodeVoter(TorchVoter):
    """Any braindecode model that takes (n_chans, n_outputs, n_times, sfreq)."""

    def _build(self, n_channels, n_times, n_classes):
        import braindecode.models as models

        return getattr(models, self.architecture)(n_chans=n_channels, n_outputs=n_classes,
                                                  n_times=n_times, sfreq=self.sfreq)


def extra_voters(sfreq: float = 128.0, epochs: int = 300) -> dict:
    voters = {
        "xdawn_tangent": make_pipeline(XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas"),
                                       TangentSpace(), StandardScaler(), _lr(0.1)),
        "erp_covariance_mdm": make_pipeline(ERPCovariances(estimator="oas", svd=4), MDM()),
        "augmented_covariance": make_pipeline(TimeDelayCovariances(sfreq), TangentSpace(),
                                              StandardScaler(), _lr(0.1)),
        "tangent_svm_8_30Hz": make_pipeline(BandCovariances(sfreq), TangentSpace(), StandardScaler(),
                                            SVC(C=1.0, probability=True, class_weight="balanced",
                                                random_state=42)),
        "bandpower_extratrees": make_pipeline(LogBandPower(sfreq),
                                              ExtraTreesClassifier(500, min_samples_leaf=2, class_weight="balanced",
                                                                   random_state=42, n_jobs=1)),
        "atcnet": BraindecodeVoter("ATCNet", sfreq, epochs=epochs),
        "fbcnet": BraindecodeVoter("FBCNet", sfreq, epochs=epochs),
        "eeginception_erp": BraindecodeVoter("EEGInceptionERP", sfreq, epochs=epochs),
    }
    return {name: clone(v) for name, v in voters.items()}


DEEP_EXTRA = {"atcnet", "fbcnet", "eeginception_erp"}
