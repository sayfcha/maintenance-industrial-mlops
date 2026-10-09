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