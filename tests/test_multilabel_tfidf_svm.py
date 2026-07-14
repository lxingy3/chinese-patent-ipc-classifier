from __future__ import annotations

import numpy as np

from experiments.multilabel_tfidf_svm import (
    choose_threshold,
    decode_scores,
    hierarchy_sets,
    micro_set_metrics,
)


def test_decode_scores_always_emits_a_label():
    scores = np.array([[-0.8, -0.2], [0.4, -0.1]])
    decoded = decode_scores(scores, threshold=0.0)

    assert decoded.tolist() == [[False, True], [True, False]]


def test_choose_threshold_uses_validation_labels():
    truth = np.array([[True, False], [False, True]])
    scores = np.array([[0.2, -0.1], [-0.2, 0.1]])
    threshold, rows = choose_threshold(truth, scores, [-0.5, 0.0, 0.5])

    assert threshold == 0.0
    assert len(rows) == 3


def test_hierarchy_metric_counts_ancestors():
    truth = [{"G06F"}, {"A01M"}]
    prediction = [{"G06N"}, {"A01M"}]
    metrics = micro_set_metrics(hierarchy_sets(truth), hierarchy_sets(prediction))

    assert metrics["precision"] == 5 / 6
    assert metrics["recall"] == 5 / 6
    assert metrics["f1"] == 5 / 6
