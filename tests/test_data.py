"""Tests for the handling of data within the model."""
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
    ex_data = load_dataset(file_name=config.app_config.test_data_file)
    return ex_data


def test_get_version():
    """Check that the version file exists and holds valid data.

    VERSION file should contain a single string in the format X.Y.Z where each
    letter corresponds to an integer number. This test checks that there are
    only three numbers in the string, and that they are correctly delimited.
    """
    assert Path("regression_model/VERSION").is_file() == True

    with open("regression_model/VERSION") as f:
        version = f.read().strip()

    assert len(version.split(".")) == 3

    for value in version.split("."):
        assert value.isnumeric() == True


def test_data_length(example_data: pd.DataFrame):
    """Check that the test dataset contains exactly ten rows."""
    assert len(example_data) == 10


def test_data_features(example_data: pd.DataFrame):
    """Check that the test dataset's columns match the config."""
    all_cols_present = True

    for field in config.app_config.imported_fields:
        if field not in example_data.columns:
            all_cols_present = False
            break

    assert all_cols_present == True


def test_get_days():
    "Test the robustness of the get_days function."
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    y_date = yesterday.strftime("%d/%m/%Y")
    assert get_days(y_date) == 1

    with pytest.raises(ValueError) as err:
        date = "01012001"
        _ = get_days(date)
    assert "Date not '/' delimited. Format should be dd/mm/yyyy" in str(err.value)

    with pytest.raises(ValueError) as err:
        date = "01/01/2001/01"
        _ = get_days(date)
    assert "Date format incorrect. Should be dd/mm/yyyy" in str(err.value)

    with pytest.raises(ValueError) as err:
        date = "32/12/2001"
        _ = get_days(date)
    assert "Day value of date must be between 1 and 31 inclusive" in str(err.value)

    with pytest.raises(ValueError) as err:
        date = "31/13/2001"
        _ = get_days(date)
    assert "Month value of date must be between 1 and 12 inclusive" in str(err.value)

    with pytest.raises(ValueError) as err:
        date = "31/12/1887"
        _ = get_days(date)
    assert "Year value of date earlier than first ever film in 1888" in str(err.value)
