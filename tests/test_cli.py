"""Tests for the CLI module."""

import os


def test_cli_group_exists() -> None:
    from llm_grammar_bench.cli import main

    assert main is not None


def test_parse_model_spec_hf_shorthand() -> None:
    """Test parsing 'hf:google/flan-t5-small'."""
    from llm_grammar_bench.cli import _parse_model_spec

    backend, model = _parse_model_spec("hf:google/flan-t5-small")
    assert backend == "huggingface"
    assert model == "google/flan-t5-small"


def test_parse_model_spec_openai() -> None:
    """Test parsing 'openai:gpt-4o'."""
    from llm_grammar_bench.cli import _parse_model_spec

    backend, model = _parse_model_spec("openai:gpt-4o")
    assert backend == "openai"
    assert model == "gpt-4o"


def test_parse_model_spec_no_prefix_defaults_to_hf() -> None:
    """Test that no prefix defaults to huggingface."""
    from llm_grammar_bench.cli import _parse_model_spec

    backend, model = _parse_model_spec("google/flan-t5-small")
    assert backend == "huggingface"
    assert model == "google/flan-t5-small"


def test_list_models_command() -> None:
    """Test that list-models command outputs available backends."""
    from click.testing import CliRunner

    from llm_grammar_bench.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["list-models"])

    assert result.exit_code == 0
    assert "openai" in result.output
    assert "anthropic" in result.output
    assert "huggingface" in result.output
    assert "openrouter" in result.output


def test_cli_loads_dotenv_from_working_directory(monkeypatch) -> None:
    """Test CLI commands load local dotenv values before running."""
    from click.testing import CliRunner

    from llm_grammar_bench.cli import main

    monkeypatch.delenv("HF_TOKEN", raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open(".env", "w") as env_file:
            env_file.write("HF_TOKEN=hf_cli_token\n")

        result = runner.invoke(main, ["list-datasets"])

    assert result.exit_code == 0
    assert result.exception is None
    assert os.environ["HF_TOKEN"] == "hf_cli_token"


def test_list_datasets_command() -> None:
    """Test that list-datasets command outputs available datasets."""
    from click.testing import CliRunner

    from llm_grammar_bench.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["list-datasets"])

    assert result.exit_code == 0
    assert "bea2019" in result.output
