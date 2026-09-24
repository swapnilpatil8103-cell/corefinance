import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from corefin.assumptions.loader import expand_series, load_config
from corefin.assumptions.schema import RootConfig
from tests.conftest import minimal_config_dict


def test_minimal_config_validates():
    config = RootConfig.model_validate(minimal_config_dict())
    assert config.timeline.n_periods == 4
    assert config.tranches[0].is_revolver


def test_fixed_rate_required_for_fixed_tranche():
    data = minimal_config_dict()
    data["tranches"][1]["rate_type"] = "fixed"
    with pytest.raises(ValidationError, match="fixed_rate required"):
        RootConfig.model_validate(data)


def test_spread_required_for_floating_tranche():
    data = minimal_config_dict()
    del data["tranches"][1]["spread"]
    with pytest.raises(ValidationError, match="spread required"):
        RootConfig.model_validate(data)


def test_sweep_priority_required_when_eligible():
    data = minimal_config_dict()
    del data["tranches"][1]["sweep_priority"]
    with pytest.raises(ValidationError, match="sweep_priority required"):
        RootConfig.model_validate(data)


def test_duplicate_sweep_priorities_rejected():
    data = minimal_config_dict()
    data["tranches"][0]["cash_sweep_eligible"] = True
    data["tranches"][0]["sweep_priority"] = 1
    with pytest.raises(ValidationError, match="sweep_priority values must be unique"):
        RootConfig.model_validate(data)


def test_commitment_fee_only_on_revolver():
    data = minimal_config_dict()
    data["tranches"][1]["commitment_fee_pct"] = 0.01
    with pytest.raises(ValidationError, match="commitment_fee_pct only applies to revolver"):
        RootConfig.model_validate(data)


def test_revolver_cannot_pik():
    data = minimal_config_dict()
    data["tranches"][0]["pik_fraction"] = 0.5
    with pytest.raises(ValidationError, match="revolver cannot have a PIK component"):
        RootConfig.model_validate(data)


def test_exit_year_index_must_be_within_timeline():
    data = minimal_config_dict(n_periods=4)
    data["transaction"]["exit_year_index"] = 10
    with pytest.raises(ValidationError, match="exit_year_index must be within"):
        RootConfig.model_validate(data)


def test_at_least_one_tranche_required():
    data = minimal_config_dict()
    data["tranches"] = []
    with pytest.raises(ValidationError, match="at least one debt tranche"):
        RootConfig.model_validate(data)


def test_load_config_from_yaml_file(tmp_path):
    data = minimal_config_dict()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, RootConfig)
    assert config.company.revenue_base_mm == 500.0


def test_expand_series_scalar_broadcasts():
    result = expand_series(0.05, n_periods=4)
    assert np.allclose(result, [0.05, 0.05, 0.05, 0.05])


def test_expand_series_list_passthrough():
    result = expand_series([0.01, 0.02, 0.03], n_periods=3)
    assert np.allclose(result, [0.01, 0.02, 0.03])


def test_expand_series_wrong_length_raises():
    with pytest.raises(ValueError, match="expected n_periods=4"):
        expand_series([0.01, 0.02], n_periods=4, field_name="revenue_growth")
