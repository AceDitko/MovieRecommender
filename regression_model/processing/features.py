"""Outline pipeline components and useful functions needed to create features."""
from typing import List

import pandas as pd
import numpy as np
import datetime
import re
from sklearn.base import BaseEstimator, TransformerMixin

from regression_model.config.core import config


class IsCinemaCreator(BaseEstimator, TransformerMixin):
    """Create Is_cinema feature."""

    def __init__(self):
        """No additional functionality required."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Return self to appease sklearn pipeline."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe to create Is_cinema feature."""
        if "Format" not in X.columns():
            raise ValueError("Is_cinema cannot be created without format col")

        # Copy the df as not to overwrite original
        X = X.copy()

        X["Is_cinema"] = X["Format"].apply(lambda x: is_cinema(x))

        return X


def IsWeekdayCreator(BaseEstimator, TransformerMixin):
    """Create Is_weekday feature."""

    def __init__(self):
        """No additional functionality required."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Return self to appease sklearn pipeline."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe to create Is_weekday feature."""
        if "Viewing Data" not in X.columns():
            raise ValueError("Is_weekday cannot be created without Viewing Date")

        # Copy the df as not to overwrite original
        X = X.copy()

        X["Is_weekday"] = X["Format"].apply(lambda x: is_weekday(x))

        return X


def BowCastCreator(BaseEstimator, TransformerMixin):
    """Create bag of words of cast and crew info feature called bow_cast."""

    def __init__(self, variables: List[str]):
        """Check that variables is of type list."""
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Return self to appease sklearn pipeline."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create the bow_cast feature."""
        # Copy the df as not to overwrite original
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].str.replace(" ", "")
            X[var] = X[var].str.replace("N/A", "nan")
            X[var] = X[var].apply(lambda x: x.lower())
            X[var] = X[var].apply(lambda x: re.split(",|/", x))

        xpld_df = pd.DataFrame(X[self.variables].values.tolist()).add_prefix("Word_")
        for col in xpld_df.columns:
            xpld_df[col] = xpld_df[col].replace("nan", np.nan)
            xpld_df[col] = xpld_df[col].fillna(value=np.nan)

        rare_encode("Word", xpld_df)

        to_join_df = xpld_df[xpld_df.columns.tolist()].values.tolist()
        to_join_df = pd.DataFrame({"bow_cast": to_join_df})

        X = X.merge(to_join_df, left_index=True, right_index=True)

        return X


def is_cinema(format):
    """Determine if film was streamed or watched in cinema."""
    if format in config.model_config.streaming_services:
        return "Stream"
    else:
        return "Cinema"


def is_weekday(date):
    """Determine if film was watched on a weekday or weekend."""
    dt_obj = datetime.datetime.strptime(date, "%d/%m/%Y")

    if dt_obj.weekday() < 5:
        return "Weekday"
    else:
        return "Weekend"


def rare_encode(self, col_str, in_df):
    """Encode infrequent variables."""
    col_str += "_"

    in_cols = [col for col in in_df.columns if col_str in col]
    temp_df = in_df[in_cols].apply(pd.Series.value_counts)
    temp_df.fillna(0, inplace=True)

    temp_df["Sum"] = 0
    print("Data frame pre count")
    print(temp_df)
    for col in in_cols:
        temp_df["Sum"] += temp_df[col]

    temp_df = temp_df[temp_df["Sum"] > 1]
    pop_vals = temp_df.index.to_list()
    for col in in_cols:
        in_df[col] = in_df[col].apply(lambda x: x if str(x) in pop_vals else "other")
