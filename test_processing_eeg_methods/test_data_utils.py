from data_utils import get_dataset_basic_info, standard_saving_path
from share import datasets_basic_infos


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
        )[-95:]
        == "voting_system_platform/Results/braincommand/processing_name/version_name_3.file_ending"
    )
