from data_utils import is_dataset_name_available, standard_saving_path
from share import datasets_basic_infos


def test_is_dataset_name_available():
    assert is_dataset_name_available(datasets_basic_infos, "braincommand") is None


def test_standard_saving_path():
    assert (
        standard_saving_path(
            datasets_basic_infos["braincommand"],
            "processing_name",
            "version_name",
            "file_ending",
            "subject_id",
        )[-95:]
        == "voting_system_platform/Results/braincommand/processing_name/version_name_subject_id.file_ending"
    )
