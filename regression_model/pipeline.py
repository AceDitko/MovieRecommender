"""Hosts the sklearn pipeline for the project."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from regression_model.config.core import config
from regression_model.processing import features as pp

numerical_transformer = SimpleImputer(missing_values=np.NaN, strategy="median")

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=np.NaN, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
