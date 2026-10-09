from pathlib import Path

import numpy as np
import pytest

from vams.data import as_text, load_emails, require_split
from vams.metrics import select_threshold
from vams.models import make_cv, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frame():
    return load_emails(ROOT / "data" / "synthetic" / "emails.csv")


def test_perfect_validation_tie_breaks_toward_one_half():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.99, 0.98, 0.02, 0.01])
    choice = select_threshold(y, p, min_precision=0.70)
    assert choice.threshold == 0.50
    assert choice.recall == 1.0
    assert choice.precision == 1.0


def test_threshold_maximizes_recall_at_precision_floor():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    p = np.array([0.90, 0.80, 0.75, 0.72, 0.71, 0.70, 0.40, 0.35, 0.33, 0.10])
    choice = select_threshold(y, p, min_precision=0.70)
    assert choice.policy == "max_recall_at_precision_0.70"
    assert choice.threshold == 0.72
    assert choice.precision >= 0.70


def test_training_is_deterministic(frame, trained_bundle):
    again = train(require_split(frame, "train"), require_split(frame, "val"), seed=42)
    bundle = trained_bundle
    probe = as_text(require_split(frame, "train").head(8))
    np.testing.assert_allclose(
        bundle.predict_action_proba(probe),
        again.predict_action_proba(probe),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        bundle.explain_estimator.named_steps["clf"].coef_,
        again.explain_estimator.named_steps["clf"].coef_,
    )
    assert bundle.threshold == again.threshold
    assert bundle.seed == 42
    assert bundle.threshold_selected_on == "validation"
    assert bundle.fitted_on == "train"
    assert set(bundle.train_ids).isdisjoint(bundle.validation_ids)
    assert set(bundle.train_template_families).isdisjoint(bundle.validation_template_families)


def test_internal_cv_keeps_template_families_disjoint(frame):
    train_frame = require_split(frame, "train")
    y = np.asarray([1 if label == "ACTION_NEEDED" else 0 for label in train_frame["label"]])
    families = train_frame["template_family"].to_numpy()
    folds = make_cv(y, families, seed=42)
    assert len(folds) == 5
    for train_index, test_index in folds:
        assert set(families[train_index]).isdisjoint(families[test_index])


def test_internal_cv_rejects_a_class_with_one_template_family():
    with pytest.raises(ValueError, match="each class needs"):
        make_cv(np.asarray([0, 0, 1, 1]), ["a", "b", "only", "only"], seed=42)


def test_internal_cv_caps_folds_by_the_smallest_class_family_count():
    y = np.asarray([0] * 4 + [1] * 8)
    families = np.asarray(
        ["negative-a"] * 2
        + ["negative-b"] * 2
        + ["positive-a"] * 2
        + ["positive-b"] * 2
        + ["positive-c"] * 2
        + ["positive-d"] * 2
    )
    folds = make_cv(y, families, seed=42)
    assert len(folds) == 2
    for train_index, test_index in folds:
        assert set(y[train_index]) == {0, 1}
        assert set(y[test_index]) == {0, 1}


def test_train_refuses_to_mix_in_test_rows(frame):
    mixed = frame.loc[frame["split"].isin(["train", "test"])]
    with pytest.raises(ValueError, match="train split"):
        train(mixed, require_split(frame, "val"), seed=42)


def test_train_refuses_cross_split_content_even_when_ids_and_families_change(frame):
    train_frame = require_split(frame, "train")
    disguised_validation = train_frame.head(2).copy()
    disguised_validation["id"] = ["other-1", "other-2"]
    disguised_validation["template_family"] = ["other-family-1", "other-family-2"]
    disguised_validation["split"] = "val"
    with pytest.raises(ValueError, match="content overlap"):
        train(train_frame, disguised_validation, seed=42)
