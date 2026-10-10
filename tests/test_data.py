import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictive_maintenance.data import split_dataset


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