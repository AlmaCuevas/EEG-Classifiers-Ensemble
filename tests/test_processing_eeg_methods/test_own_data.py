import warnings

import numpy as np
import pandas as pd
import pytest

from processing_eeg_methods.ensemble import run_own_data
from processing_eeg_methods.ensemble.reproduce import check_fingerprints
from processing_eeg_methods.ensemble.voter_sets import (
    FAST,
    V1,
    V2,
    get_voters,
    voter_names,
)

warnings.filterwarnings("ignore")
VOTERS = "covariance_tangent_space,riemann_mdm_8_30Hz,erp_time_course"
LABELS = np.array(["left", "right", "up"])


def recording(n_trials=96, n_channels=4, sfreq=256.0, seed=0):
    """Two sessions; channel c has more power when the trial's class is c."""
    rng = np.random.default_rng(seed)
    y = LABELS[rng.integers(0, len(LABELS), n_trials)]
    X = rng.standard_normal((n_trials, n_channels, int(sfreq)))
    for c, label in enumerate(LABELS):
        X[y == label, c] *= 2.0
    session = np.repeat([0, 1], n_trials // 2)
    run = np.tile(np.repeat(np.arange(4), n_trials // 8), 2)
    return X, y, sfreq, session, run


def test_npz_dataset_end_to_end(tmp_path):
    folder = tmp_path / "data" / "toy"
    folder.mkdir(parents=True)
    for sid in (1, 2):
        X, y, sfreq, session, run = recording(seed=sid)
        np.savez(
            folder / f"subject_{sid}.npz",
            X=X,
            y=y,
            sfreq=sfreq,
            session=session,
            run=run,
        )
    df = run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "out")
    )
    assert len(df) == 2
    assert (df["ensemble: auto"] > 0.6).all()  # chance is 0.33
    assert (tmp_path / "out" / "toy" / "custom" / "per_subject.csv").exists()
    saved = np.load(tmp_path / "out" / "toy" / "custom" / "subject_1.npz")
    assert set(saved["classes"]) == set(LABELS)
    assert saved["y_test"].shape[0] == 48  # trained on session 0, tested on session 1


def test_resumes_without_retraining(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    X, y, sfreq, session, run = recording()
    np.savez(folder / "subject_1.npz", X=X, y=y, sfreq=sfreq, session=session, run=run)
    first = run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path), out_dir=str(tmp_path / "out")
    )
    second = run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path), out_dir=str(tmp_path / "out")
    )
    pd.testing.assert_frame_equal(first, second)


def test_single_session_without_runs_splits_in_halves(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    X, y, sfreq, _, _ = recording()
    np.savez(folder / "subject_a.npz", X=X, y=y, sfreq=sfreq)
    run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path), out_dir=str(tmp_path / "out")
    )
    saved = np.load(tmp_path / "out" / "toy" / "custom" / "subject_a.npz")
    assert saved["y_test"].shape[0] == 48 and saved["y_oof"].shape[0] == 48


def test_mne_epochs_file(tmp_path):
    mne = pytest.importorskip("mne")
    X, y, sfreq, session, run = recording()
    info = mne.create_info([f"C{i}" for i in range(X.shape[1])], sfreq, "eeg")
    codes = {label: i + 1 for i, label in enumerate(LABELS)}
    events = np.c_[
        np.arange(len(y)) * 300, np.zeros(len(y), int), [codes[v] for v in y]
    ]
    epochs = mne.EpochsArray(
        X * 1e-6,
        info,
        events,
        event_id=codes,
        verbose="error",
        metadata=pd.DataFrame({"session": session, "run": run}),
    )
    (tmp_path / "toy").mkdir()
    epochs.save(tmp_path / "toy" / "subject_1-epo.fif", verbose="error")
    df = run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path), out_dir=str(tmp_path / "out")
    )
    assert df["ensemble: auto"].iloc[0] > 0.6


def test_missing_class_in_test_session_is_excluded(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    X, y, sfreq, session, run = recording()
    y[session == 1] = "left"
    np.savez(folder / "subject_1.npz", X=X, y=y, sfreq=sfreq, session=session, run=run)
    df = run_own_data.run(
        "toy", VOTERS, data_dir=str(tmp_path), out_dir=str(tmp_path / "out")
    )
    assert df.empty


def test_voter_sets_match_the_paper():
    assert len(V1) == 12 and len(V2) == 15 and len(FAST) == 10
    assert voter_names("v1") == V1 and voter_names("a,b") == ["a", "b"]
    assert set(get_voters("fast")) == set(FAST)  # needs the [features] extra


def test_frozen_code_matches_protocol_fingerprints():
    assert check_fingerprints()
