"""The voter sets evaluated in the paper, by the paper's names.

  main       the 12 voters of the main ensemble (experiment 1; PROTOCOL.md "v1",
             frozen 2026-10-23)
  curated    15 of 20 voters, chosen by backward elimination on development data only
             (experiment 2; PROTOCOL.md "v2", Amendment A1, frozen 2026-10-27)
  classical  the curated voters that are not networks: no GPU needed, calibrates in minutes
             (PROTOCOL.md "fast")

The PROTOCOL.md names v1, v2 and fast are accepted as aliases.

    from processing_eeg_methods.ensemble.voter_sets import get_voters
    ens = VotingEnsemble(get_voters("classical", sfreq=128.0)).fit(X, y, groups=runs)
"""

from processing_eeg_methods.ensemble.extra_voters import DEEP_EXTRA, extra_voters
from processing_eeg_methods.ensemble.voters import deep_voters, default_voters

V1 = [
    "coherence",
    "complexity_boosting",
    "covariance_tangent_space",
    "covariance_tangent_space_4_30Hz",
    "csp_lda_8_30Hz",
    "eegnet",
    "erp_time_course",
    "green",
    "minirocket",
    "relative_spectrum",
    "riemann_mdm_8_30Hz",
    "shallownet",
]
V2 = [
    "atcnet",
    "augmented_covariance",
    "coherence",
    "complexity_boosting",
    "covariance_tangent_space",
    "eegnet",
    "erp_covariance_mdm",
    "erp_time_course",
    "fbcnet",
    "green",
    "minirocket",
    "relative_spectrum",
    "shallownet",
    "tangent_svm_8_30Hz",
    "xdawn_tangent",
]
DEEP = {"eegnet", "shallownet", "green"} | DEEP_EXTRA
FAST = [v for v in V2 if v not in DEEP]
SETS = {"main": V1, "curated": V2, "classical": FAST}
ALIASES = {
    "v1": "main",
    "v2": "curated",
    "fast": "classical",
}  # names used in PROTOCOL.md
MAIN, CURATED, CLASSICAL = V1, V2, FAST


def voter_names(spec):
    """'main' | 'curated' | 'classical' (or 'v1' | 'v2' | 'fast') | a list (or comma-separated
    string) of voter names."""
    if isinstance(spec, str):
        spec = SETS.get(ALIASES.get(spec, spec), spec.split(","))
    return list(spec)


def voter_pool(sfreq=128.0, epochs=300, deep=True):
    """Every voter used anywhere in the benchmark (20), built for this sampling rate."""
    pool = {**default_voters(sfreq), **extra_voters(sfreq, epochs=epochs)}
    if deep:
        pool.update(deep_voters(sfreq, epochs=epochs))
    else:
        pool = {k: v for k, v in pool.items() if k not in DEEP}
    return pool


def get_voters(spec="classical", sfreq=128.0, epochs=300):
    """Voters by set name or list of names. Raises if one cannot be built here (missing package)."""
    names = voter_names(spec)
    pool = voter_pool(sfreq, epochs, deep=bool(DEEP & set(names)))
    missing = [n for n in names if n not in pool]
    if missing:
        raise ImportError(
            f"cannot build voters {missing}: install the optional dependencies "
            "(pip install -e .[all]) and, for 'green', run tools/install_green.py"
        )
    return {n: pool[n] for n in names}
