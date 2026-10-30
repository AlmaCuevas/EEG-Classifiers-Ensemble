"""Voters for the VotingEnsemble.

A voter is any scikit-learn classifier with predict_proba that takes epoched EEG
(trials, channels, samples). default_voters() returns a deliberately diverse set:
different signal aspects (spatial covariance, time course, spectrum, complexity,
connectivity, raw-shape kernels) and different learners (linear, distance-to-mean,
gradient boosting), so that their errors are as independent as possible.
"""

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from processing_eeg_methods.ensemble.features import (
    CoherenceFeatures,
    ComplexityFeatures,
    ERPFeatures,
    FilterBankTangentSpace,
    RelativeSpectrum,
    bandpass,
)


class BandCovariances(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=250.0, l_freq=8.0, h_freq=30.0):
        self.sfreq, self.l_freq, self.h_freq = sfreq, l_freq, h_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Covariances("oas").transform(bandpass(X, self.l_freq, self.h_freq, self.sfreq))


class MiniRocketVoter(BaseEstimator, ClassifierMixin):
    """MiniRocket random convolution kernels + logistic regression.
    Requires the optional package ``sktime``."""

    def __init__(self, sfreq=250.0, num_kernels=5000, C=0.001):
        self.sfreq, self.num_kernels, self.C = sfreq, num_kernels, C

    def _prep(self, X):
        return bandpass(X, 1, 40, self.sfreq).astype(np.float32)

    def fit(self, X, y):
        from sktime.transformations.panel.rocket import MiniRocketMultivariate

        self.rocket_ = MiniRocketMultivariate(num_kernels=self.num_kernels, random_state=42)
        feats = self.rocket_.fit_transform(self._prep(X)).to_numpy()
        self.clf_ = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced"),
        ).fit(feats, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.rocket_.transform(self._prep(X)).to_numpy())

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class TorchVoter(BaseEstimator, ClassifierMixin):
    """Wraps a PyTorch EEG network as a voter. Trains for a fixed number of epochs
    (no early stopping on held-out data, so nothing leaks from the test fold)."""

    def __init__(self, architecture="EEGNet", sfreq=250.0, epochs=300, lr=1e-3,
                 batch_size=32, device=None, seed=42):
        self.architecture, self.sfreq, self.epochs, self.lr = architecture, sfreq, epochs, lr
        self.batch_size, self.device, self.seed = batch_size, device, seed

    def _build(self, n_channels, n_times, n_classes):
        if self.architecture == "GREEN":
            import os
            import sys

            from processing_eeg_methods.share import ROOT_VOTING_SYSTEM_PATH

            sys.path.insert(0, ROOT_VOTING_SYSTEM_PATH)  # neuro_green lives at the repo root
            from neuro_green.green.research_code.pl_utils import get_green
            from neuro_green.green.wavelet_layers import RealCovariance

            return get_green(n_freqs=8, kernel_width_s=0.5, conv_stride=4, oct_min=0, oct_max=5.5,
                             n_ch=n_channels, sfreq=self.sfreq, dropout=0.5, hidden_dim=[32],
                             pool_layer=RealCovariance(), bi_out=[min(16, n_channels)],
                             out_dim=n_classes, orth_weights=True)
        from braindecode.models import EEGNetv4, ShallowFBCSPNet

        if self.architecture == "EEGNet":
            return EEGNetv4(n_channels, n_classes, n_times=n_times) if _new_braindecode() else \
                EEGNetv4(n_channels, n_classes, input_window_samples=n_times)
        if self.architecture == "ShallowNet":
            kw = {"n_times": n_times} if _new_braindecode() else {"input_window_samples": n_times}
            return ShallowFBCSPNet(n_channels, n_classes, final_conv_length="auto", **kw)
        raise ValueError(self.architecture)

    def _prep(self, X):
        x = bandpass(X, 1, min(45, 0.45 * self.sfreq), self.sfreq)
        x = (x - x.mean(-1, keepdims=True)) / self.scale_
        return x.astype(np.float32)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.scale_ = 1.0
        xb = bandpass(X, 1, min(45, 0.45 * self.sfreq), self.sfreq)
        self.scale_ = float(np.std(xb)) + 1e-9
        xt = torch.tensor(self._prep(X), device=device)
        yt = torch.tensor(self.encoder_.transform(y), dtype=torch.long, device=device)
        self.model_ = self._build(X.shape[1], X.shape[2], len(self.classes_)).to(device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.model_(xt[idx]), yt[idx]).backward()
                opt.step()
            sched.step()
        self.device_ = device
        del xt, yt
        return self

    def predict_proba(self, X):
        import torch

        self.model_.eval()
        x = torch.tensor(self._prep(X), device=self.device_)
        with torch.no_grad():
            out = torch.cat([torch.softmax(self.model_(x[i : i + 256]), 1) for i in range(0, len(x), 256)])
        return out.cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _new_braindecode():
    import braindecode

    return int(braindecode.__version__.split(".")[0]) >= 1


class CSPVoter(BaseEstimator, ClassifierMixin):
    """Common Spatial Patterns + LDA, the classic motor-imagery pipeline (needs mne)."""

    def __init__(self, sfreq=250.0, l_freq=8.0, h_freq=30.0, n_components=6):
        self.sfreq, self.l_freq, self.h_freq, self.n_components = sfreq, l_freq, h_freq, n_components

    def fit(self, X, y):
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.model_ = make_pipeline(CSP(n_components=self.n_components, reg="ledoit_wolf", log=True),
                                    LinearDiscriminantAnalysis())
        self.model_.fit(bandpass(X, self.l_freq, self.h_freq, self.sfreq), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(bandpass(X, self.l_freq, self.h_freq, self.sfreq))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class LegacyMethodVoter(BaseEstimator, ClassifierMixin):
    """Adapter so the repository's original ProcessingMethod classes (spatial_features,
    ShallowFBCSPNet, LSTM, ...) can take part in the VotingEnsemble."""

    def __init__(self, method_factory=None, dataset_info=None, subject_id=0):
        self.method_factory, self.dataset_info, self.subject_id = method_factory, dataset_info, subject_id

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.method_ = self.method_factory()
        self.method_.train(subject_id=self.subject_id, data=X, labels=self.encoder_.transform(y),
                           dataset_info=self.dataset_info)
        return self

    def predict_proba(self, X):
        probs = [np.ravel(self.method_.test(subject_id=self.subject_id, data=X[i : i + 1],
                                            dataset_info=self.dataset_info))[: len(self.classes_)]
                 for i in range(len(X))]
        probs = np.clip(np.nan_to_num(np.array(probs, dtype=float)), 0, None) + 1e-9
        return probs / probs.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _lr(C):
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


def deep_voters(sfreq: float = 250.0, epochs: int = 300) -> dict:
    """GPU-friendly network voters: EEGNet, ShallowNet (braindecode) and GREEN (neuro_green,
    needs geotorch + lightning). Skipped individually if their dependencies are missing."""
    voters = {}
    try:
        import braindecode  # noqa: F401

        voters["eegnet"] = TorchVoter("EEGNet", sfreq, epochs=epochs)
        voters["shallownet"] = TorchVoter("ShallowNet", sfreq, epochs=epochs)
    except ImportError:
        pass
    try:
        import geotorch  # noqa: F401
        import lightning  # noqa: F401

        voters["green"] = TorchVoter("GREEN", sfreq, epochs=epochs, lr=3e-3)
    except ImportError:
        pass
    return voters


def default_voters(sfreq: float = 250.0, include_optional: bool = True) -> dict:
    """Diverse default voters. Optional ones (antropy, sktime, mne) are skipped if missing."""
    voters = {
        "covariance_tangent_space": make_pipeline(
            FilterBankTangentSpace(sfreq), StandardScaler(), _lr(0.01)),
        "covariance_tangent_space_4_30Hz": make_pipeline(
            FilterBankTangentSpace(sfreq, bands=((4, 8), (8, 13), (13, 20), (20, 30))),
            StandardScaler(), _lr(0.1)),
        "erp_time_course": make_pipeline(ERPFeatures(sfreq), StandardScaler(), _lr(0.01)),
        "relative_spectrum": make_pipeline(RelativeSpectrum(sfreq), StandardScaler(), _lr(0.01)),
        "coherence": make_pipeline(CoherenceFeatures(sfreq), StandardScaler(), _lr(0.01)),
        "riemann_mdm_8_30Hz": make_pipeline(BandCovariances(sfreq), MDM()),
    }
    if include_optional:
        try:
            import antropy  # noqa: F401

            voters["complexity_boosting"] = make_pipeline(
                ComplexityFeatures(sfreq),
                HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=8,
                                               min_samples_leaf=10, class_weight="balanced",
                                               random_state=42))
        except ImportError:
            pass
        try:
            import sktime  # noqa: F401

            voters["minirocket"] = MiniRocketVoter(sfreq)
        except ImportError:
            pass
        try:
            import mne  # noqa: F401

            voters["csp_lda_8_30Hz"] = CSPVoter(sfreq)
        except ImportError:
            pass
    return {name: clone(v) for name, v in voters.items()}
