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


def test_quarterly_period_length_is_a_quarter_year():
    tl = Timeline.quarterly(n_periods=9)
    assert np.allclose(tl.period_length_years, 0.25)
    assert tl.n_periods == 9
    assert np.all(tl.is_projection)


def test_quarterly_with_historical():
    tl = Timeline.quarterly(n_periods=6, n_historical=2)
    assert list(tl.is_projection) == [False, False, True, True, True, True]
    assert tl.n_projection_periods == 4


def test_quarterly_year_labels_roll_over_correctly():
    tl = Timeline.quarterly(n_periods=6, start_year=2007, start_quarter=3)
    assert tl.year_labels == ["2007Q3", "2007Q4", "2008Q1", "2008Q2", "2008Q3", "2008Q4"]


def test_quarterly_year_labels_default_to_q1():
    tl = Timeline.quarterly(n_periods=2, start_year=2020)
    assert tl.year_labels == ["2020Q1", "2020Q2"]


def test_quarterly_without_start_year_uses_period_labels():
    tl = Timeline.quarterly(n_periods=2)
    assert tl.start_quarter is None
    assert tl.year_labels == ["Period 1", "Period 2"]


def test_quarterly_rejects_bad_start_quarter():
    with pytest.raises(ValueError, match="start_quarter"):
        Timeline.quarterly(n_periods=4, start_year=2020, start_quarter=5)


def test_start_quarter_requires_start_year():
    with pytest.raises(ValueError, match="start_quarter requires start_year"):
        Timeline(
            n_periods=3,
            period_length_years=np.full(3, 0.25),
            is_projection=np.ones(3, dtype=bool),
            start_quarter=1,
        )
