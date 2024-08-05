import time
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import pandas as pd
from classifiers_classes import (
    GRU_function,
    LSTM_function,
    ProcessingMethod,
    ShallowFBCSPNet_function,
    customized_function,
    diffE_function,
    feature_extraction_function,
    selected_transformers_function,
)


@dataclass
class probability_input:
    trial_group_index: int
    group_index: int
    dataset_name: str
    methods: str
    probabilities: List[float]
    subject_id: int
    channel: int
    kfold: int
    label: int
    training_accuracy: float
    training_timing: float
    testing_timing: float


@dataclass
class complete_experiment:
    data_point: List[probability_input] = field(default_factory=list)

    def to_df(self):
        return pd.DataFrame(self.data_point)


@dataclass
class ModelPerformance:
    accuracy: float
    timing: float


@dataclass
class SingleOutput:
    probabilities: List[Any]
    timing: float


@dataclass
class MethodInfo:
    activation: bool
    function: ProcessingMethod
    training: ModelPerformance
    testing: SingleOutput


@dataclass
class ProcessingMethods:
    selected_transformers: MethodInfo = field(
        init=False
    )  # Training is over-fitted. Training accuracy >90
    customized: MethodInfo = field(
        init=False
    )  # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
    ShallowFBCSPNet: MethodInfo = field(init=False)
    LSTM: MethodInfo = field(
        init=False
    )  # Training is over-fitted. Training accuracy >90
    GRU: MethodInfo = field(
        init=False
    )  # Training is over-fitted. Training accuracy >90
    diffE: MethodInfo = field(
        init=False
    )  # It doesn't work if you only use one channel in the data
    feature_extraction: MethodInfo = field(init=False)

    def activate_methods(
        self,
        selected_transformers: bool,
        customized: bool,
        ShallowFBCSPNet: bool,
        LSTM: bool,
        GRU: bool,
        diffE: bool,
        feature_extraction: bool,
        number_of_classes: int,
    ):
        self.selected_transformers = MethodInfo(
            activation=selected_transformers,
            function=selected_transformers_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )  # note: I didn't use the Optional because the values inside where duplicating as if they were the same variable.
        self.customized = MethodInfo(
            activation=customized,
            function=customized_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )
        self.ShallowFBCSPNet = MethodInfo(
            activation=ShallowFBCSPNet,
            function=ShallowFBCSPNet_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )
        self.LSTM = MethodInfo(
            activation=LSTM,
            function=LSTM_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )
        self.GRU = MethodInfo(
            activation=GRU,
            function=GRU_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )
        self.diffE = MethodInfo(
            activation=diffE,
            function=diffE_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )
        self.feature_extraction = MethodInfo(
            activation=feature_extraction,
            function=feature_extraction_function(),
            training=ModelPerformance(accuracy=np.nan, timing=np.nan),
            testing=SingleOutput(
                probabilities=[[np.nan] * number_of_classes], timing=np.nan
            ),
        )

    def get_activated_methods(self):
        activated_methods = []
        for method_name in vars(self):
            method = getattr(self, method_name)
            if method.activation:
                activated_methods.append(method_name)
        return activated_methods

    def train(self, subject_id: int, data, labels, dataset_info: dict):

        for method_name in vars(self):
            method = getattr(self, method_name)
            if method.activation:
                print(f"Training {method_name}...")
                start_time = time.time()
                method.training.accuracy = method.function.train(
                    subject_id=subject_id,
                    data=data,
                    labels=labels,
                    dataset_info=dataset_info,
                )  # todo: Training accuracies are not reliable (its in reality a mini-testing inside the training), therefore it would be better to stop getting them and focus all the samples into pure training
                method.training.timing = time.time() - start_time

    def test(self, subject_id: int, data, dataset_info: dict):
        """
        Returns
        -------
        Final list of probabilities, where each number represents each class.
        This list is the summary from all models, the ensemble model.
        """

        for method_name in vars(self):
            method = getattr(self, method_name)
            if method.activation:
                print(f"Testing {method_name}...")
                start_time = time.time()
                method.testing.probabilities = method.function.test(
                    subject_id=subject_id, data=data, dataset_info=dataset_info
                )
                method.testing.timing = time.time() - start_time

    def voting_decision(
        self,  # Ensemble in real time
        voting_by_mode: bool = False,
        weighted_accuracy: bool = True,
    ):
        ensemble_probabilities_summary = []
        if voting_by_mode:
            for method_name in vars(self):
                method = getattr(self, method_name)
                if method.training.accuracy is not np.nan:
                    ensemble_probabilities_summary.append(
                        np.argmax(method.testing.probabilities)
                    )
            return ensemble_probabilities_summary
        else:  # voting by array of probabilities
            probs_list = []
            if weighted_accuracy:
                for method_name in vars(self):
                    method = getattr(self, method_name)
                    probs_list.append(
                        np.multiply(
                            method.testing.probabilities, method.training.accuracy
                        )
                    )
            else:
                for method_name in vars(self):
                    method = getattr(self, method_name)
                    probs_list.append(method.testing.probabilities)

            ensemble_probabilities_summary = np.nanmean(
                probs_list, axis=0
            )  # Mean over columns

            return ensemble_probabilities_summary
