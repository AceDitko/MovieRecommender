"""Tests for the handling of new features within the model."""
import sys
from pathlib import Path
import pytest

# Add this because it's bad to have an __init__.py in tests folder
sys.path.append(str(Path(__file__).resolve().parent.parent))

from regression_model.config.core import config
from regression_model.processing.data_manager import *


@pytest.fixture()
def example_data():
    """Create/load the test dataset."""
    ex_data = load_dataset(file_Name=config.app_config.test_data_file)
    return ex_data
