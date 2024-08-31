import mne
import pandas as pd
from mne.decoding import CSP
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from processing_eeg_methods.data_utils import (
    ClfSwitcher,
    get_best_classificator_and_test_accuracy,
)

# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb


def get_spatial_and_frequency_features_data(
    data, dataset_info: dict, labels=None, transform_methods: dict = {}
) -> tuple[pd.DataFrame, dict]:
    features: dict = {
        # Do not use 'Vect' transform, most of the time is nan or 0.25 if anything.
        "ERPcova": Pipeline(
            [("ERPcova", ERPCovariances(estimator="oas")), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
        "XdawnCova": Pipeline(
            [("XdawnCova", XdawnCovariances(estimator="oas")), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
        "CSP": Pipeline(
            [("Vectorizer", CSP(n_components=4, reg=None, log=True, norm_trace=False))]
        ),
        "Cova": Pipeline(
            [("Cova", Covariances()), ("ts", TangentSpace())]
        ),  # Add TangentSpace, otherwise the dimensions are not 2D.
    }
    frequency_ranges: dict = {
        "complete": [0, int(dataset_info["sample_rate"] / 2) - 1],
        "delta": [0, 3],
        "theta": [3, 7],
        "alpha": [7, 13],
        "beta 1": [13, 16],
        "beta 2": [16, 20],
        "beta 3": [20, 35],
        "gamma": [35, int(dataset_info["sample_rate"] / 2) - 1],
    }

    features_df = pd.DataFrame()

    for feature_name, feature_method in features.items():
        if labels is not None:
            transform_methods[feature_name] = Pipeline([(feature_name, feature_method)])
        for frequency_bandwidth_name, frequency_bandwidth in frequency_ranges.items():
            print(frequency_bandwidth)
            iir_params = dict(order=8, ftype="butter")
            filt = mne.filter.create_filter(
                data,
                dataset_info["sample_rate"],
                l_freq=frequency_bandwidth[0],
                h_freq=frequency_bandwidth[1],
                method="iir",
                iir_params=iir_params,
                verbose=False,
            )
            filtered = signal.sosfiltfilt(filt["sos"], data)

            if labels is not None:
                X_features = transform_methods[feature_name].fit_transform(
                    filtered, labels
                )
            else:
                X_features = transform_methods[feature_name].transform(filtered)
            print("Combined space has", X_features.shape[1], "features")
            column_name = [
                f"{frequency_bandwidth_name}_{feature_name}_{num}"
                for num in range(0, X_features.shape[1])
            ]
            temp_features_df = pd.DataFrame(X_features, columns=column_name)
            features_df = pd.concat([features_df, temp_features_df], axis=1)
    return features_df, transform_methods


def spatial_features_train(features_df, labels):
    X_SelectKBest = SelectKBest(f_classif, k=100)
    X_new = X_SelectKBest.fit_transform(features_df, labels)
    columns_list = X_SelectKBest.get_feature_names_out()
    features_df = pd.DataFrame(X_new, columns=columns_list)

    classifier, acc = get_best_classificator_and_test_accuracy(
        features_df, labels, Pipeline([("clf", ClfSwitcher())])
    )
    return classifier, acc, columns_list


def spatial_features_test(clf, features_df):
    """
    This is what the real-time BCI will call.
    Parameters
    ----------
    clf : classifier trained for the specific subject
    features_df: one epoch, the current one that represents the intention of movement of the user.

    Returns Array of classification with 4 floats representing the target classification
    -------

    """

    array = clf.predict_proba(features_df)

    return array
