import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

import joblib
import numpy as np

from processing_eeg_methods.BigProject.GRU_probs import GRU_test, GRU_train
from processing_eeg_methods.BigProject.LSTM_probs import LSTM_test, LSTM_train
from processing_eeg_methods.data_utils import (
    convert_into_binary,
    convert_into_independent_channels,
    data_normalization,
)
from processing_eeg_methods.diffusion.diffE_probs import diffE_test
from processing_eeg_methods.diffusion.diffE_training import diffE_train
from processing_eeg_methods.NeuroTechX_dl_eeg.ShallowFBCSPNet_probs import (
    ShallowFBCSPNet_test,
    ShallowFBCSPNet_train,
)
from processing_eeg_methods.spatial_features.simplified_spatial_features_probs import (
    simplified_spatial_features_test,
    simplified_spatial_features_train,
)
from processing_eeg_methods.spatial_features.spatial_features_probs import (
    get_spatial_and_frequency_features_data,
    spatial_features_test,
    spatial_features_train,
)
from processing_eeg_methods.time_features.time_features_probs import (
    by_frequency_band,
    extractions_test,
    extractions_train,
)


class ProcessingMethod:
    """
    Provides a generic definition of the processing methods to create and use them.
    """

    def train(self, **kwargs) -> float:
        """
        Returns the accuracy
        """
        raise Exception("Not implemented yet.")

    def test(self, **kwargs) -> List[float]:
        """
        Returns the probabilities
        """
        raise Exception("Not implemented yet.")

    def save(self, path, method_name: str):
        for name, attribute in self.__dict__.items():
            file_name = f"{method_name}_{name}.joblib"
            print("/".join((path, file_name)))
            with open("/".join((path, file_name)), "wb") as f:
                joblib.dump(attribute, f)

    @classmethod
    def load(cls, path, method_name: str):
        my_model = {}
        for name in cls.__annotations__:
            file_name = f"{method_name}_{name}.joblib"
            print("/".join((path, file_name)))
            with open("/".join((path, file_name)), "rb") as f:
                my_model[name] = joblib.load(f)
        return cls(**my_model)


@dataclass
class spatial_features_function(ProcessingMethod):
    clf: Optional[Any] = None
    columns_list: List[str] = field(default_factory=list)
    transform_methods: dict = field(default_factory=dict)

    def train(self, data, labels, dataset_info: dict, **kwargs):
        features_train_df, self.transform_methods = (
            get_spatial_and_frequency_features_data(
                data, dataset_info=dataset_info, labels=labels
            )
        )
        (
            self.clf,
            accuracy,
            self.columns_list,
        ) = spatial_features_train(features_train_df, labels)
        return accuracy

    def test(self, data, dataset_info: dict, **kwargs):
        transforms_test_df, _ = get_spatial_and_frequency_features_data(
            data,
            dataset_info=dataset_info,
            labels=None,
            transform_methods=self.transform_methods,
        )
        return data_normalization(
            spatial_features_test(self.clf, transforms_test_df[self.columns_list])
        )


@dataclass
class simplified_spatial_features_function(ProcessingMethod):
    clf: Optional[Any] = None

    def train(self, data, labels, **kwargs):
        (
            self.clf,
            accuracy,
            processing_name,
        ) = simplified_spatial_features_train(copy.deepcopy(data), labels)
        return accuracy

    def test(self, data, **kwargs):
        return data_normalization(simplified_spatial_features_test(self.clf, data))


@dataclass
class ShallowFBCSPNet_function(ProcessingMethod):

    def train(self, data, labels, dataset_info: dict, subject_id: int, **kwargs):
        temp_data = (data * 1e6).astype(np.float32)
        model_ShallowFBCSPNet_accuracies = []
        for chosen_numbered_label in range(0, dataset_info["#_class"] + 1):
            temp_labels = convert_into_binary(
                labels.copy(), chosen_numbered_label=chosen_numbered_label
            )
            model_ShallowFBCSPNet_accuracies.append(
                ShallowFBCSPNet_train(
                    temp_data,
                    temp_labels,
                    chosen_numbered_label=chosen_numbered_label,
                    dataset_info=dataset_info,
                    subject_id=subject_id,
                )
            )
        return np.mean(model_ShallowFBCSPNet_accuracies)

    def test(self, data, dataset_info: dict, subject_id: int, **kwargs):
        temp_data_array = (data * 1e6).astype(np.float32)
        ShallowFBCSPNet_arrays = []
        for chosen_numbered_label in range(0, dataset_info["#_class"]):
            ShallowFBCSPNet_arrays.append(
                ShallowFBCSPNet_test(
                    subject_id,
                    temp_data_array,
                    dataset_info,
                    chosen_numbered_label=chosen_numbered_label,
                )[0]
            )
        return data_normalization(
            np.array([prob_array[1] for prob_array in ShallowFBCSPNet_arrays]).reshape(
                1, -1
            )
        )


@dataclass
class LSTM_function(ProcessingMethod):
    clf: Optional[Any] = None

    def train(self, data, labels, dataset_info: dict, subject_id: int, **kwargs):
        self.clf, accuracy = LSTM_train(dataset_info, data, labels, subject_id)
        return accuracy

    def test(self, data, **kwargs):
        return data_normalization(LSTM_test(self.clf, data))


@dataclass
class GRU_function(ProcessingMethod):
    clf: Optional[Any] = None

    def train(self, data, labels, dataset_info: dict, subject_id: int, **kwargs):
        self.clf, accuracy = GRU_train(
            dataset_info["dataset_name"], data, labels, dataset_info["#_class"]
        )
        return accuracy

    def test(self, data, **kwargs):
        return data_normalization(GRU_test(self.clf, data))


@dataclass
class diffE_function(ProcessingMethod):

    def train(self, data, labels, dataset_info: dict, subject_id: int, **kwargs):
        return diffE_train(
            subject_id=subject_id, X=data, Y=labels, dataset_info=dataset_info
        )  # The trained clf is saved in a file

    def test(self, data, dataset_info: dict, subject_id: int, **kwargs):
        return data_normalization(
            diffE_test(subject_id=subject_id, X=data, dataset_info=dataset_info)
        )


@dataclass
class feature_extraction_function(ProcessingMethod):
    clf: Optional[Any] = None

    def train(self, data, labels, dataset_info: dict, subject_id: int, **kwargs):
        data_simplified, labels_simplified = convert_into_independent_channels(
            data, labels
        )
        features_df = by_frequency_band(data_simplified, dataset_info)
        (
            self.clf,
            accuracy,
        ) = extractions_train(features_df, labels_simplified)
        return accuracy

    def test(self, data, dataset_info: dict, subject_id: int, **kwargs):
        data_array_simplified, _ = convert_into_independent_channels(data, [1])
        features_df = by_frequency_band(data_array_simplified, dataset_info)
        return data_normalization(extractions_test(self.clf, features_df))
