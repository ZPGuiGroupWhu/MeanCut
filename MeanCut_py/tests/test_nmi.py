import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score

from MeanCut_py.meancut_core import metrics


@pytest.mark.parametrize(
    ("reference", "predicted"),
    [
        ([0, 0, 1, 1], [1, 1, 2, 2]),
        ([0, 0, 1, 1], [0, 0, 1, 1]),
        ([1, 1, 2, 2], [2, 2, 1, 1]),
        ([-4, -4, 9, 9], [12, 12, -6, -6]),
        ([3, 3, 11, 11], [17, 17, 2, 2]),
        ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2]),
        ([1, 1, 2, 2, 3, 3], [1, 2, 2, 2, 3, 3]),
        ([-2, -2, -2, 4, 4, 4], [0, 0, 3, 3, 3, 5]),
        ([-2, -2, -2, -2, -2, -2], [0, 0, 3, 3, 3, 5]),
    ],
)
def test_nmi_agrees_with_arithmetic_reference(reference, predicted):
    reference = np.array(reference, dtype=np.int64)
    predicted = np.array(predicted, dtype=np.int64)
    expected = normalized_mutual_info_score(reference, predicted, average_method="arithmetic")

    assert metrics.get_nmi(reference, predicted) == pytest.approx(expected)
    assert metrics.get_nmi(predicted, reference) == pytest.approx(expected)
    assert metrics.clust_eval(reference, predicted)[1] == pytest.approx(expected)


def test_sparse_integer_ids_do_not_size_arrays_by_label(monkeypatch):
    reference = np.array([10**9, 10**9, -10**9, -10**9])
    predicted = np.array([-7, -7, 10**9, 10**9])
    original_zeros = np.zeros

    def checked_zeros(size, *args, **kwargs):
        assert np.prod(size) <= reference.size, "NMI allocated by label magnitude"
        return original_zeros(size, *args, **kwargs)

    monkeypatch.setattr(metrics.np, "zeros", checked_zeros)

    assert metrics.get_nmi(reference, predicted) == pytest.approx(1.0)


def test_nmi_is_unchanged_by_sample_order_and_does_not_mutate_inputs():
    rng = np.random.default_rng(253)
    reference = np.tile([-3, 0, -3, 4, 4, 9, 9, 9], 512)
    predicted = np.tile([2, 2, -4, 5, 5, 0, 2, 0], 512)
    original_reference, original_predicted = reference.copy(), predicted.copy()
    order = rng.permutation(reference.size)
    expected = normalized_mutual_info_score(reference, predicted, average_method="arithmetic")

    assert metrics.get_nmi(reference, predicted) == pytest.approx(expected)
    assert metrics.get_nmi(reference[order], predicted[order]) == pytest.approx(expected)
    np.testing.assert_array_equal(reference, original_reference)
    np.testing.assert_array_equal(predicted, original_predicted)


@pytest.mark.parametrize(("reference", "predicted"), [([], []), ([0], [-7])])
def test_empty_and_singleton_conventions_are_api_specific(reference, predicted):
    assert metrics.get_nmi(reference, predicted) == 0.0
    assert metrics.clust_eval(reference, predicted) == (1.0,) * 6


def test_constant_partitions_keep_zero_nmi():
    assert metrics.get_nmi([0, 0, 0], [-7, -7, -7]) == 0.0
    assert metrics.clust_eval([0, 0, 0], [-7, -7, -7])[1] == 0.0


def test_inputs_are_flattened_and_mismatched_lengths_still_error():
    assert metrics.get_nmi([[1, 1], [2, 2]], [2, 2, 1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.get_nmi([1, 2], [1, 2, 1])
    with pytest.raises(ValueError, match="same length"):
        metrics.clust_eval([1, 2], [1, 2, 1])


def test_other_clustering_scores_keep_their_values_and_return_order():
    relabelings = [
        ([1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 2, 3]),
        ([-2, -2, -2, 4, 4, 4], [0, 0, 3, 3, 3, 5]),
    ]
    for reference, predicted in relabelings:
        acc, _, adjusted_rand, f_score, jaccard, rand = metrics.clust_eval(reference, predicted)
        # There are two same/same, four split, two merged, and seven different/different pairs.
        assert (acc, adjusted_rand, f_score, jaccard, rand) == pytest.approx(
            (2 / 3, 2 / 17, 2 / 5, 1 / 4, 3 / 5)
        )


def test_cli_scores_zero_based_labels_from_dataset(tmp_path):
    features = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]])
    reference = np.repeat([0, 1], 4)
    dataset = tmp_path / "two clusters.txt"
    np.savetxt(dataset, np.column_stack((np.vstack((features, features + 10)), reference)))
    script = Path(__file__).resolve().parents[1] / "main.py"

    result = subprocess.run(
        [sys.executable, "-B", str(script), "--dataset", str(dataset), "--params", "k2=3"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    scores = {
        key.strip(): float(value)
        for key, value in (line.split(":", 1) for line in result.stdout.splitlines()[2:])
    }
    assert scores == {name: 1.0 for name in ("ACC", "NMI", "ARI", "Fscore", "JI", "RI")}
