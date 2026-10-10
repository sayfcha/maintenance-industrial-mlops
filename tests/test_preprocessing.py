import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictive_maintenance.preprocessing import (
    NUMERIC_FEATURES,
    build_preprocessor,
)


@pytest.fixture
def sample_inputs():
    return pd.DataFrame({
        "Air temperature [K]": [298.0, 300.0, 302.0],
        "Process temperature [K]": [308.0, 310.0, 312.0],
        "Rotational speed [rpm]": [1400, 1500, 1600],
        "Torque [Nm]": [30.0, 40.0, 50.0],
        "Tool wear [min]": [0, 100, 200],
        "Type": ["H", "L", "M"],
    })

def test_preprocessing_excludes_nonfeatures(sample_inputs):
    extra_columns = {
        "UDI": [1, 2, 3],
        "Product ID": ["H001", "L002", "M003"],
        "Machine failure": [0, 1, 0],
        "TWF": [0, 1, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 0],
        "RNF": [0, 0, 0],
    }
    data_with_extras = sample_inputs.assign(**extra_columns)

    preprocessor = build_preprocessor()
    expected = preprocessor.fit_transform(sample_inputs)
    actual = preprocessor.transform(data_with_extras)

    assert actual.shape == (3, 8)
    assert_frame_equal(actual, expected)

def test_transform_preserves_training_statistics(sample_inputs):
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_inputs)

    scaler = preprocessor.named_transformers_["numeric"]
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()

    validation = sample_inputs.copy()
    validation["Air temperature [K]"] += 10

    transformed = preprocessor.transform(validation)

    assert scaler.mean_ == pytest.approx(original_mean)
    assert scaler.scale_ == pytest.approx(original_scale)

    expected_first_temperature = (
        validation["Air temperature [K]"].iloc[0] - original_mean[0]
    ) / original_scale[0]

    assert transformed["numeric__Air temperature [K]"].iloc[0] == (
        pytest.approx(expected_first_temperature)
    )

def test_preprocessing_rejects_unknown_category(sample_inputs):
    preprocessor = build_preprocessor()
    preprocessor.fit(sample_inputs)

    invalid_data = sample_inputs.copy()
    invalid_data.loc[0, "Type"] = "Z"

    with pytest.raises(ValueError, match="unknown categories"):
        preprocessor.transform(invalid_data)