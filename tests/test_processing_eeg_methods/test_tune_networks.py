import pandas as pd

from processing_eeg_methods.ensemble.tune_networks import CONFIGS, choose


def scores(values):
    """values: {(paradigm, network, config, dataset): [oof_bacc per subject]}"""
    rows = [
        {
            "paradigm": p,
            "network": n,
            "config": c,
            "dataset": d,
            "subject": s,
            "oof_bacc": v,
        }
        for (p, n, c, d), subject_values in values.items()
        for s, v in enumerate(subject_values)
    ]
    return pd.DataFrame(rows)


def test_datasets_weigh_equally_whatever_their_size():
    # dataset A (4 subjects) favours "paper"; dataset B (1 subject) favours "native" more strongly
    df = scores(
        {
            ("MI", "eegnet", "paper", "A"): [0.70] * 4,
            ("MI", "eegnet", "native_input_paper_training", "A"): [0.66] * 4,
            ("MI", "eegnet", "paper", "B"): [0.50],
            ("MI", "eegnet", "native_input_paper_training", "B"): [0.60],
        }
    )
    selected, table = choose(df)
    # dataset means: paper (0.70 + 0.50) / 2 = 0.60 < native (0.66 + 0.60) / 2 = 0.63
    assert selected == {"MI": {"eegnet": "native_input_paper_training"}}
    assert len(table) == 2


def test_ties_go_to_the_earlier_cheaper_configuration():
    configs = list(CONFIGS)
    df = scores({("P300", "green", c, "A"): [0.8, 0.6] for c in reversed(configs)})
    assert choose(df)[0] == {"P300": {"green": configs[0]}}


def test_each_paradigm_and_network_is_chosen_separately():
    df = scores(
        {
            ("MI", "atcnet", "paper", "A"): [0.6],
            ("MI", "atcnet", "native_input_published_training", "A"): [0.7],
            ("SSVEP", "atcnet", "paper", "B"): [0.6],
            ("SSVEP", "atcnet", "native_input_published_training", "B"): [0.5],
        }
    )
    assert choose(df)[0] == {
        "MI": {"atcnet": "native_input_published_training"},
        "SSVEP": {"atcnet": "paper"},
    }
