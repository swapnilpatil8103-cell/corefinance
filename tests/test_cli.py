from pathlib import Path

import yaml
from typer.testing import CliRunner

from corefin.cli import app
from tests.test_debt_integration import debt_config_dict

runner = CliRunner()

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "example_midmarket.yaml"


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_run_command_against_example_config(tmp_path):
    output_path = tmp_path / "out.xlsx"
    result = runner.invoke(
        app, ["run", "--config", str(EXAMPLE_CONFIG), "--output", str(output_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "Sources & Uses" in result.stdout
    assert "Sponsor returns" in result.stdout
    assert output_path.exists()


def test_run_command_missing_config_fails_cleanly():
    result = runner.invoke(app, ["run", "--config", "does_not_exist.yaml"])
    assert result.exit_code != 0


def test_optimize_command_against_example_config(tmp_path):
    with EXAMPLE_CONFIG.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Shrink the search for test speed; still exercises the full CLI path
    # (grid search, refinement, confirmation, Excel export, heatmap PNG).
    data["optimizer"]["search"]["grid_points_per_dimension"] = 4
    data["optimizer"]["search"]["refine_grid_points_per_dimension"] = 3
    data["optimizer"]["search"]["n_scenarios_search"] = 200
    data["optimizer"]["search"]["n_scenarios_confirm"] = 400
    small_config_path = tmp_path / "small_optimizer_config.yaml"
    with small_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    output_path = tmp_path / "optimization.xlsx"
    heatmap_path = tmp_path / "heatmap.png"
    result = runner.invoke(
        app,
        [
            "optimize",
            "--config",
            str(small_config_path),
            "--output",
            str(output_path),
            "--heatmap",
            str(heatmap_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Recommended structure" in result.stdout
    assert "Binding constraints" in result.stdout
    assert "Comparison vs input config structure" in result.stdout
    assert output_path.exists()
    assert heatmap_path.exists()


def test_optimize_command_requires_optimizer_section(tmp_path):
    data = debt_config_dict(n_periods=4)
    config_path = tmp_path / "no_optimizer.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    result = runner.invoke(app, ["optimize", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "optimizer" in result.output.lower()
