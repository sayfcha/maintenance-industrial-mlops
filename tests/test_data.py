import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictive_maintenance.data import split_dataset, validate_raw_data


@pytest.fixture
def sample_data():
    """Create a small dataset with unique IDs and 4% failures."""
    return pd.DataFrame({
        "UDI": range(1, 1001),
        "Machine failure": [0] * 960 + [1] * 40,
    })

def test_split_preserves_rows_and_sizes(sample_data):
    train, validation, test = split_dataset(sample_data)

    assert (len(train), len(validation), len(test)) == (600, 200, 200)

    reconstructed = (
        pd.concat([train, validation, test])
        .sort_values("UDI")
        .reset_index(drop=True)
    )

    expected = sample_data.sort_values("UDI").reset_index(drop=True)

    assert_frame_equal(reconstructed, expected)


def test_split_preserves_failure_rate(sample_data):
    splits = split_dataset(sample_data)

    for split in splits:
        failure_rate = split["Machine failure"].mean()
        assert failure_rate == pytest.approx(0.04)


def test_split_is_reproducible(sample_data):
    first_run = split_dataset(sample_data, random_state=42)
    second_run = split_dataset(sample_data, random_state=42)

    for first_split, second_split in zip(first_run, second_run):
        assert_frame_equal(first_split, second_split)


@pytest.fixture
def valid_raw_data():
    """Create two observations satisfying the raw-data schema."""
    return pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L00001", "M00002"],
        "Type": ["L", "M"],
        "Air temperature [K]": [300.0, 301.0],
        "Process temperature [K]": [310.0, 311.0],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 45.0],
        "Tool wear [min]": [10, 220],
        "Machine failure": [0, 1],
        "TWF": [0, 1],
        "HDF": [0, 0],
        "PWF": [0, 0],
        "OSF": [0, 0],
        "RNF": [0, 0],
    })

def test_validation_accepts_valid_data(valid_raw_data):
    validate_raw_data(valid_raw_data)


def test_validation_rejects_missing_column(valid_raw_data):
    invalid_data = valid_raw_data.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(invalid_data)


def test_validation_rejects_empty_dataset(valid_raw_data):
    invalid_data = valid_raw_data.iloc[:0]

    with pytest.raises(ValueError, match="must contain observations"):
        validate_raw_data(invalid_data)


def test_validation_rejects_missing_value(valid_raw_data):
    valid_raw_data.loc[0, "Torque [Nm]"] = float("nan")

    with pytest.raises(ValueError, match="Missing values"):
        validate_raw_data(valid_raw_data)


def test_validation_rejects_duplicate_ids(valid_raw_data):
    valid_raw_data.loc[1, "UDI"] = valid_raw_data.loc[0, "UDI"]

    with pytest.raises(ValueError, match="UDI values must be unique"):
        validate_raw_data(valid_raw_data)


def test_validation_rejects_unknown_type(valid_raw_data):
    valid_raw_data.loc[0, "Type"] = "Z"

    with pytest.raises(ValueError, match="Type must contain only"):
        validate_raw_data(valid_raw_data)


def test_validation_rejects_nonbinary_target(valid_raw_data):
    valid_raw_data.loc[0, "Machine failure"] = 2

    with pytest.raises(ValueError, match="Machine failure must contain only"):
        validate_raw_data(valid_raw_data)


def test_validation_rejects_duplicate_columns(valid_raw_data):
    invalid_data = valid_raw_data.rename(columns={"Type": "UDI"})

    with pytest.raises(ValueError, match="Column names must be unique"):
        validate_raw_data(invalid_data)