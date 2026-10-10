from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictive_maintenance.preprocessing import build_preprocessor


def build_baseline_pipelines() -> dict[str, Pipeline]:
    """Build fresh, unfitted pipelines for the initial benchmark."""
    return {
        "dummy": Pipeline([
            ("preprocessing", build_preprocessor()),
            ("model", DummyClassifier(strategy="constant", constant=0)),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessing", build_preprocessor()),
            ("model", LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=1000,
            )),
        ]),
    }