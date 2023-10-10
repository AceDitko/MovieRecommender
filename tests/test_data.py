"""Tests for the handling of data within the model."""
import sys
from pathlib import Path
import pytest

# Add this because it's bad to have an __init__.py in tests folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from regression_model.config.core import config
from regression_model.processing.data_manager import *


# @pytest.fixture()
def test_input_data():
    """Create/load the test dataset."""
    test_df = load_dataset(file_name=config.app_config.test_data_file)
    print(len(test_df))
    assert len(test_df) == 10
