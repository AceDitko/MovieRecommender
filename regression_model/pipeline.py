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

feature_creator = Pipeline(
    steps=[
        ("IsCinemaCreator", pp.IsCinemaCreator()),
        ("IsWeekdayCreator", pp.IsWeekdayCreator()),
        (
            "BowCastCreator",
            pp.BowCastCreator(variables=config.app_config.cast_and_crew_vars),
        ),
        ("TfidfCreator", pp.TfidfCreator(config.app_config.tfidf_vars)),
    ]
)

rate_pipe = Pipeline(
    steps=[
        numerical_transformer,
        categorical_transformer,
        feature_creator,
        ("XGBoostHandler", pp.XgboostHandler),
    ]
)
