import pytest
from data_classification import probabilities_to_answer, voting_decision
from numpy import array, array_equal, nan


@pytest.fixture
def methods():
    return {
        "ShallowFBCSPNet": True,
        "feature_extraction": True,
        "selected_transformers": True,
    }


@pytest.mark.parametrize(
    ("models_outputs", "voting_by_mode", "answer"),
    (
        pytest.param(
            {
                "selected_transformers_accuracy": 0.66,
                "ShallowFBCSPNet_accuracy": nan,
                "feature_extraction_accuracy": 0.52,
                "selected_transformers_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "ShallowFBCSPNet_probabilities": array([[nan, nan, nan, nan]]),
                "feature_extraction_probabilities": array(
                    [[0.49750067, 0.5414916, 0.50335746, 0.45377439]]
                ),
            },
            True,
            [1, 1],
            id="By mode, 2 methods",
        ),
        pytest.param(
            {
                "selected_transformers_accuracy": 0.66,
                "ShallowFBCSPNet_accuracy": 0.71,
                "feature_extraction_accuracy": 0.52,
                "selected_transformers_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "ShallowFBCSPNet_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "feature_extraction_probabilities": array(
                    [[0.49750067, 0.5414916, 0.50335746, 0.45377439]]
                ),
            },
            True,
            [1, 1, 1],
            id="By mode, 3 methods",
        ),
        pytest.param(
            {
                "selected_transformers_accuracy": 0.66,
                "ShallowFBCSPNet_accuracy": nan,
                "feature_extraction_accuracy": 0.52,
                "selected_transformers_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "ShallowFBCSPNet_probabilities": array([[nan, nan, nan, nan]]),
                "feature_extraction_probabilities": array(
                    [[0.49750067, 0.5414916, 0.50335746, 0.45377439]]
                ),
            },
            False,
            array([[0.3060453841, 0.4064967771, 0.146391975, 0.20066574870000004]]),
            id="By probability average, 2 methods",
        ),
        pytest.param(
            {
                "selected_transformers_accuracy": 0.66,
                "ShallowFBCSPNet_accuracy": 0.71,
                "feature_extraction_accuracy": 0.52,
                "selected_transformers_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "ShallowFBCSPNet_probabilities": array(
                    [[0.53544003, 0.80517867, 0.04702738, 0.25055881]]
                ),
                "feature_extraction_probabilities": array(
                    [[0.49750067, 0.5414916, 0.50335746, 0.45377439]]
                ),
            },
            False,
            array(
                [
                    [
                        0.33075106316666664,
                        0.4615568033,
                        0.10872446326666667,
                        0.1930760841666667,
                    ]
                ]
            ),
            id="By probability average, 3 methods",
        ),
    ),
)
def test_voting_decision(methods, models_outputs, voting_by_mode, answer):
    assert array_equal(voting_decision(methods, models_outputs, voting_by_mode), answer)


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
