import pytest
from data_utils import (
    get_dataset_basic_info,
    probabilities_to_answer,
    standard_saving_path,
)
from numpy import array
from share import datasets_basic_infos


@pytest.mark.parametrize(
    ("probs_by_channels", "voting_by_mode", "answer"),
    (
        pytest.param(
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            True,
            1,
            id="By mode, 2 methods",
        ),
        pytest.param(
            [
                [0, 3, 1],
                [2, 1, 1],
                [1, 2, 0],
                [0, 3, 0],
                [0, 3, 2],
                [0, 1, 2],
                [3, 2, 1],
                [2, 2, 1],
            ],
            True,
            1,
            id="By mode, 3 methods",
        ),
        pytest.param(
            [
                array([[0.32513345, 0.3945682, 0.14108255, 0.20406704]]),
                array([[0.29374468, 0.41149468, 0.14975986, 0.21587926]]),
                array([[0.28999249, 0.41386514, 0.14796669, 0.21585909]]),
                array([[0.30545609, 0.40742003, 0.14825152, 0.20653249]]),
                array([[0.29859713, 0.40770251, 0.14513981, 0.21876508]]),
                array([[0.29370486, 0.41322397, 0.14697595, 0.21155912]]),
                array([[0.31822545, 0.39956075, 0.14730608, 0.20366565]]),
                array([[0.30712263, 0.40818219, 0.14634092, 0.20109982]]),
            ],
            False,
            1,
            id="By probability average, any number of methods",
        ),
    ),
)
def test_probabilities_to_answer(probs_by_channels, voting_by_mode, answer):
    assert probabilities_to_answer(probs_by_channels, voting_by_mode) == answer


def test_is_dataset_name_available():
    datasets_basic_info = get_dataset_basic_info(datasets_basic_infos, "braincommand")
    assert isinstance(datasets_basic_info, dict)


def test_standard_saving_path():
    assert (
        standard_saving_path(
            datasets_basic_infos["braincommand"],
            "processing_name",
            "version_name",
            "file_ending",
            3,
        )[-63:]
        == "Results/braincommand/processing_name/version_name_3.file_ending"
    )
