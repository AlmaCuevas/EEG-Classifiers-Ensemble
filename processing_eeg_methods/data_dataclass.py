from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class probability_input:
    methods: str
    probabilities: List[int]
    subject_id: int
    channel: int
    kfold: int
    label: int


@dataclass
class complete_experiment:
    data_point: List[probability_input] = field(default_factory=list)

    def to_df(self):
        return pd.DataFrame(self.data_point)
