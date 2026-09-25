import numpy as np

from corefin.assumptions.schema import RootConfig
from corefin.transaction.returns import compute_exit_and_returns, vectorized_irr
from corefin.transaction.sources_uses import compute_sources_and_uses
from tests.test_debt_integration import debt_config_dict


def test_sources_and_uses_arithmetic():
    config = RootConfig.model_validate(debt_config_dict(n_periods=5))
    sau = compute_sources_and_uses(
        config.transaction,
        config.tranches,
        config.company.revenue_base_mm,
        entry_ebitda_margin=0.20,
    )
    assert sau.entry_ebitda_mm == 500.0 * 0.20
    assert sau.purchase_price_mm == 8.0 * sau.entry_ebitda_mm
    assert sau.transaction_fees_mm == 0.0
    assert sau.financing_fees_and_oid_mm == 0.0
    assert sau.total_debt_sources_mm == 250.0 + 100.0
    assert sau.sponsor_equity_mm == sau.total_uses_mm - sau.total_debt_sources_mm
    np.testing.assert_allclose(sau.sponsor_equity_mm, sau.purchase_price_mm - 350.0)


def test_sources_and_uses_includes_fees_and_oid():
    data = debt_config_dict(n_periods=5)
    data["transaction"]["transaction_fees_pct"] = 0.02
    data["tranches"][1]["upfront_fee_pct"] = 0.01
    data["tranches"][1]["oid_pct"] = 0.005
    config = RootConfig.model_validate(data)
    sau = compute_sources_and_uses(
        config.transaction,
        config.tranches,
        config.company.revenue_base_mm,
        entry_ebitda_margin=0.20,
    )
    expected_financing_fees = 0.015 * 250.0
    assert np.isclose(sau.financing_fees_and_oid_mm, expected_financing_fees)
    expected_fees = 0.02 * sau.purchase_price_mm
    assert np.isclose(sau.transaction_fees_mm, expected_fees)
    expected_uses = sau.purchase_price_mm + expected_fees + expected_financing_fees
    assert np.isclose(sau.total_uses_mm, expected_uses)


def test_vectorized_irr_known_case():
    # invest 100, receive 200 five years later -> IRR = 2**(1/5) - 1
    cash_flows = np.array([[-100.0, 0.0, 0.0, 0.0, 0.0, 200.0]])
    irr = vectorized_irr(cash_flows)
    expected = 2.0 ** (1.0 / 5.0) - 1.0
    np.testing.assert_allclose(irr, [expected], atol=1e-6)


def test_vectorized_irr_matches_moic_power_rule_no_interim_flows():
    n_years = 4
    moics = np.array([1.5, 2.0, 3.0, 0.5])
    cash_flows = np.zeros((4, n_years + 1))
    cash_flows[:, 0] = -100.0
    cash_flows[:, -1] = 100.0 * moics
    irr = vectorized_irr(cash_flows)
    expected = moics ** (1.0 / n_years) - 1.0
    np.testing.assert_allclose(irr, expected, atol=1e-6)


def test_exit_and_returns_moic_no_dividends():
    exit_year_index = 3
    n_scenarios = 3
    ebitda = np.zeros((n_scenarios, exit_year_index + 1))
    ebitda[:, exit_year_index] = [100.0, 120.0, 80.0]
    total_debt = np.zeros_like(ebitda)
    total_debt[:, exit_year_index] = [200.0, 150.0, 250.0]
    cash = np.zeros_like(ebitda)
    cash[:, exit_year_index] = [20.0, 30.0, 10.0]
    dividends = np.zeros_like(ebitda)
    exit_multiple = np.array([8.0, 8.0, 8.0])
    sponsor_equity_mm = 150.0

    result = compute_exit_and_returns(
        sponsor_equity_mm, exit_year_index, exit_multiple, ebitda, total_debt, cash, dividends
    )

    expected_enterprise_value = exit_multiple * ebitda[:, exit_year_index]
    expected_net_debt = total_debt[:, exit_year_index] - cash[:, exit_year_index]
    expected_equity_value = expected_enterprise_value - expected_net_debt
    np.testing.assert_allclose(result.exit_equity_value, expected_equity_value)

    expected_moic = expected_equity_value / sponsor_equity_mm
    np.testing.assert_allclose(result.moic, expected_moic)

    # exit_year_index is 0-indexed, so exiting at the end of that period means
    # exit_year_index + 1 full years have elapsed since entry (t=0).
    expected_irr = expected_moic ** (1.0 / (exit_year_index + 1)) - 1.0
    np.testing.assert_allclose(result.irr, expected_irr, atol=1e-6)


def test_exit_and_returns_moic_includes_interim_dividends():
    exit_year_index = 2
    ebitda = np.array([[0.0, 0.0, 100.0]])
    total_debt = np.array([[0.0, 0.0, 100.0]])
    cash = np.array([[0.0, 0.0, 10.0]])
    dividends = np.array([[5.0, 5.0, 0.0]])
    exit_multiple = np.array([8.0])
    sponsor_equity_mm = 100.0

    result = compute_exit_and_returns(
        sponsor_equity_mm, exit_year_index, exit_multiple, ebitda, total_debt, cash, dividends
    )
    exit_equity_value = 8.0 * 100.0 - (100.0 - 10.0)
    expected_moic = (5.0 + 5.0 + exit_equity_value) / sponsor_equity_mm
    np.testing.assert_allclose(result.moic, [expected_moic])


def test_exit_equity_value_floored_at_zero_when_underwater():
    # Net debt (300) exceeds enterprise value (8x * 20 = 160): a wipeout.
    # Limited liability means the sponsor's realized proceeds and MOIC can't
    # go negative, even though the raw enterprise-value-minus-net-debt does.
    exit_year_index = 1
    ebitda = np.array([[0.0, 20.0]])
    total_debt = np.array([[0.0, 320.0]])
    cash = np.array([[0.0, 20.0]])
    dividends = np.zeros_like(ebitda)
    exit_multiple = np.array([8.0])
    sponsor_equity_mm = 100.0

    result = compute_exit_and_returns(
        sponsor_equity_mm, exit_year_index, exit_multiple, ebitda, total_debt, cash, dividends
    )

    assert result.exit_equity_value[0] < 0
    assert result.realized_exit_equity_value[0] == 0.0
    assert result.moic[0] == 0.0
    # A total wipeout is close to -100% IRR (the solver clips at -99.9999% to
    # keep (1+r) away from zero), not undefined or a positive/absurd value.
    assert result.irr[0] < -0.99


def test_moic_never_negative_across_a_range_of_leverage_outcomes():
    # A grid of increasingly underwater exits: MOIC should floor at 0, never go
    # negative, regardless of how far net debt exceeds enterprise value.
    exit_year_index = 0
    ebitda = np.array([[10.0]] * 5)
    total_debt = np.array([[50.0], [100.0], [200.0], [500.0], [1000.0]])
    cash = np.zeros((5, 1))
    dividends = np.zeros((5, 1))
    exit_multiple = np.full(5, 8.0)
    sponsor_equity_mm = 50.0

    result = compute_exit_and_returns(
        sponsor_equity_mm, exit_year_index, exit_multiple, ebitda, total_debt, cash, dividends
    )
    assert np.all(result.moic >= 0.0)
