import pytest

from predictive_maintenance.evaluation import evaluate_predictions


@pytest.mark.parametrize(
    "threshold, counts, precision, recall, f1, cost",
    [
        (0.5, (1, 1, 1, 1), 0.5, 0.5, 0.5, 12_000),
        (0.3, (1, 1, 0, 2), 2 / 3, 1.0, 0.8, 3_000),
    ],
)
def test_evaluation_matches_hand_calculated_results(
    threshold, counts, precision, recall, f1, cost
):
    result = evaluate_predictions(
        [0, 0, 1, 1],
        [0.1, 0.7, 0.4, 0.9],
        threshold=threshold,
    )

    assert tuple(result[key] for key in ("tn", "fp", "fn", "tp")) == counts
    assert result["precision"] == pytest.approx(precision)
    assert result["recall"] == pytest.approx(recall)
    assert result["f1"] == pytest.approx(f1)
    assert result["business_cost_eur"] == cost

    # Ranking metrics do not depend on the decision threshold.
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "invalid_probability, message",
    [
        (-0.1, "between 0 and 1"),
        (1.1, "between 0 and 1"),
        (float("nan"), "finite"),
    ],
)
def test_evaluation_rejects_invalid_probabilities(
    invalid_probability, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(
            [0, 1],
            [0.2, invalid_probability],
        )