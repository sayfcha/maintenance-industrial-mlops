import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test splits with a 60/20/20 ratio.

    Preserve approximate failure prevalence through stratification.
    The input must contain the binary target 'Machine failure'.
    """
    target = "Machine failure"

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df[target],
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df[target],
        random_state=random_state,
    )

    return train_df, val_df, test_df

def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the structure and labels of the raw AI4I dataset."""
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    label_columns = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    required_columns = [
        "UDI",
        "Product ID",
        "Type",
        *numeric_features,
        *label_columns,
    ]

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("The dataset must contain observations.")

    columns_with_missing = (
        df[required_columns].isna().any()
    )
    if columns_with_missing.any():
        names = columns_with_missing[columns_with_missing].index.tolist()
        raise ValueError(f"Missing values in columns: {names}")

    if not df["UDI"].is_unique:
        raise ValueError("UDI values must be unique.")

    if not df["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must contain only L, M, or H.")

    for column in label_columns:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")