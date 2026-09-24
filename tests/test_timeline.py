import numpy as np
import pytest

from corefin.timeline import Timeline


def test_annual_all_projection():
    tl = Timeline.annual(n_periods=7)
    assert tl.n_periods == 7
    assert np.all(tl.is_projection)
    assert tl.n_projection_periods == 7
    assert np.allclose(tl.period_length_years, 1.0)


def test_annual_with_historical():
    tl = Timeline.annual(n_periods=5, n_historical=2)
    assert list(tl.is_projection) == [False, False, True, True, True]
    assert tl.n_projection_periods == 3


def test_year_labels_with_start_year():
    tl = Timeline.annual(n_periods=3, start_year=2024)
    assert tl.year_labels == ["2024", "2025", "2026"]


def test_year_labels_without_start_year():
    tl = Timeline.annual(n_periods=2)
    assert tl.year_labels == ["Period 1", "Period 2"]


def test_rejects_bad_n_historical():
    with pytest.raises(ValueError):
        Timeline.annual(n_periods=3, n_historical=4)


def test_rejects_non_positive_n_periods():
    with pytest.raises(ValueError):
        Timeline.annual(n_periods=0)


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Timeline(
            n_periods=3,
            period_length_years=np.ones(2),
            is_projection=np.ones(3, dtype=bool),
        )
