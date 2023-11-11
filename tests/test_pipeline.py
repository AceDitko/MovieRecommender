"""Tests for the handling of pipelines within the model."""
import sys
from pathlib import Path
import pytest

from sklearn import train_test_split

# Add this because it's bad to have an __init__.py in tests folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from regression_model.config.core import config
from regression_model.processing.data_manager import *


@pytest.fixture()
def example_data():
    """Create/load the test dataset."""
    ex_data = load_dataset(file_Name=config.app_config.test_data_file)
    return ex_data


# def test_pipeline():
#    """Execute the pipeline with the test dataset."""
#    X_train, X_val, y_train, y_val = train_test_split(example_data[config.app_config.features], example_data['True Rating'])
