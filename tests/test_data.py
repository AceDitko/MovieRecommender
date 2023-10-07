"""Tests for the handling of data within the model."""
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from regression_model.config.core import config
from regression_model.processing.data_manager import *


@pytest.fixture()
def simple_input_data():
    """Create/load the test dataset."""
    return load_dataset(file_name=config.app_config.test_data_file)
