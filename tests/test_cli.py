from pathlib import Path

from typer.testing import CliRunner

from corefin.cli import app

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
