from experiments.run_large_rag_eval import (
    classification_metrics,
    expected_is_hallucinated,
    prediction_is_hallucinated,
)


def test_expected_label_mapping():
    assert expected_is_hallucinated("supported") is False
    assert expected_is_hallucinated("unsupported") is True
    assert expected_is_hallucinated("hallucinated_answer") is True


def test_prediction_mapping_and_classification_metrics():
    assert prediction_is_hallucinated("supported", 0.0) is False
    assert prediction_is_hallucinated("partially_supported", 0.5) is True
    assert prediction_is_hallucinated("unsupported", 0.0) is True

    metrics = classification_metrics(
        expected=[True, True, False, False],
        predicted=[True, False, True, False],
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
