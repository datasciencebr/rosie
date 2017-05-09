
from click.testing import CliRunner

from rosie.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.output == """Usage: root [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  chamber_of_deputies  serenata data path
"""
    assert result.exit_code == 0
