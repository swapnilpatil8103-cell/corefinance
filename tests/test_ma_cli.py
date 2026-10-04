"""Offline tests for corefin.ma.cli -- synthetic data only, no real FFIEC/
FR Y-9C/credit-engine files (those are gitignored and not available in
CI; `run`'s own real-data path is exercised manually, not here)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from corefin.credit.interface import CreditLossProjection
from corefin.ma.cli import (
    BankIdentity,
    ExampleDealConfig,
    LocalBankIdentityOverride,
    _balance_weighted_total_nco_rate,
    app,
    load_example_deal_config,
)
from corefin.timeline import Timeline

runner = CliRunner()
EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "example_bank_deal.yaml"


def test_load_example_deal_config_parses_the_committed_example():
    example = load_example_deal_config(EXAMPLE_CONFIG)
    assert example.acquirer.label == "Acquirer Bank A"
    assert example.target.label == "Target Bank B"
    assert example.deal.consideration.price_to_tbv == pytest.approx(1.5)
    assert example.deal.consideration.stock_pct == pytest.approx(1.0)
    assert example.deal.credit_mark.credit_mark_pct == pytest.approx(0.02)
    assert example.deal.credit_mark.pcd_share == pytest.approx(0.15)
    assert example.deal.cdi.cdi_pct_of_core_deposits == pytest.approx(0.015)
    assert example.deal.cost_saves.cost_save_pct_of_target_noninterest_expense == pytest.approx(
        0.25
    )
    assert example.deal.cost_saves.restructuring_charge_mm == pytest.approx(40.0)
    assert example.deal.irr.exit_multiple == pytest.approx(1.5)
    assert example.deal.deal_horizon_quarters == 20


def test_load_example_deal_config_local_override_replaces_acquirer_and_target(tmp_path):
    local_path = tmp_path / "local.yaml"
    local_path.write_text(
        "acquirer:\n"
        '  label: "My Acquirer"\n'
        '  bank_id: "11111"\n'
        '  hc_rssd_id: "22222"\n'
        "target:\n"
        '  label: "My Target"\n'
        '  bank_id: "33333"\n'
        '  hc_rssd_id: "44444"\n'
    )
    example = load_example_deal_config(EXAMPLE_CONFIG, local_path)
    assert example.acquirer.label == "My Acquirer"
    assert example.acquirer.bank_id == "11111"
    assert example.target.bank_id == "33333"
    # the deal TERMS still come from the main --config, unaffected by the local override
    assert example.deal.consideration.price_to_tbv == pytest.approx(1.5)


def test_load_example_deal_config_without_local_override_keeps_main_config():
    example = load_example_deal_config(EXAMPLE_CONFIG, None)
    assert example.acquirer.label == "Acquirer Bank A"


def test_local_bank_identity_override_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LocalBankIdentityOverride.model_validate(
            {
                "acquirer": {"label": "A", "bank_id": "1", "hc_rssd_id": "2"},
                "target": {"label": "B", "bank_id": "3", "hc_rssd_id": "4"},
                "deal": {"not_allowed_here": 1},
            }
        )


def test_bank_identity_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        BankIdentity(label="x", bank_id="1", hc_rssd_id="2", not_a_real_field=1)


def test_example_deal_config_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        ExampleDealConfig.model_validate(
            {
                "acquirer": {"label": "A", "bank_id": "1", "hc_rssd_id": "2"},
                "target": {"label": "B", "bank_id": "3", "hc_rssd_id": "4"},
                "deal": {
                    "consideration": {
                        "price_to_tbv": 1.5,
                        "stock_pct": 1.0,
                        "acquirer_share_price": 10.0,
                        "acquirer_shares_outstanding_mm": 10.0,
                    },
                    "credit_mark": {"credit_mark_pct": 0.02, "pcd_share": 0.15},
                },
                "not_a_real_field": 1,
            }
        )


def test_run_command_missing_config_fails_cleanly():
    result = runner.invoke(app, ["run", "--config", "does_not_exist.yaml"])
    assert result.exit_code != 0


def test_run_command_missing_local_config_fails_cleanly():
    result = runner.invoke(
        app,
        ["run", "--config", str(EXAMPLE_CONFIG), "--local-config", "does_not_exist.yaml"],
    )
    assert result.exit_code != 0


def test_run_command_missing_data_file_fails_cleanly_not_a_traceback(tmp_path):
    # a VALID config, pointing at data files that don't exist -- should hit _require_file's
    # own clean error, not an opaque pandas/numpy exception deep in the pipeline.
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(EXAMPLE_CONFIG),
            "--call-report-q4",
            str(tmp_path / "nope.zip"),
        ],
    )
    assert result.exit_code != 0
    assert "Error" in result.output  # typer's own BadParameter panel, not a raw traceback
    assert result.exception is None or isinstance(result.exception, SystemExit)


_TIMELINE = Timeline.quarterly(3, n_historical=1, start_year=2025, start_quarter=4)
_CATEGORIES = ("commercial_and_industrial", "residential_mortgage")


def test_balance_weighted_total_nco_rate_weights_by_jumpoff_balance():
    credit_proj = CreditLossProjection(
        timeline=_TIMELINE,
        categories=_CATEGORIES,
        scenario_name="baseline",
        bank_identifier="111",
        balance_mm=np.array([[800.0, 800.0, 800.0], [200.0, 200.0, 200.0]]),
        net_charge_off_mm=np.full((2, 3), np.nan),
        provision_expense_mm=np.full((2, 3), np.nan),
        allowance_mm=np.full((2, 3), 10.0),
        npl_mm=np.full((2, 3), 5.0),
        nco_rate=np.array([[np.nan, 0.03, 0.03], [np.nan, 0.08, 0.08]]),
        npl_ratio=np.full((2, 3), 0.01),
        monte_carlo_mean=np.zeros(2),
        monte_carlo_percentiles={},
    )
    training = pd.DataFrame(
        {
            "bank_id": ["111", "111"],
            "category": list(_CATEGORIES),
            "quarter": [pd.Period("2025Q4", freq="Q")] * 2,
            "winsorized_nco_rate": [0.01, 0.05],
        }
    )
    realized, projected = _balance_weighted_total_nco_rate(
        credit_proj, training, "111", pd.Period("2025Q4", freq="Q")
    )
    assert realized == pytest.approx((0.01 * 800.0 + 0.05 * 200.0) / 1000.0)
    assert projected == pytest.approx((0.03 * 800.0 + 0.08 * 200.0) / 1000.0)
