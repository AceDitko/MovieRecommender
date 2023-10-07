"""Tests for the handling of data within the model."""
import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import *


@pytest.fixture()
def simple_input_data():
    """Create/load the test dataset."""
    return load_dataset(file_name=config.app_config.test_data_file)
