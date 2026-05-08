# AGENTS.md — LLM Grammar Bench

## Tooling (mandatory)

This project enforces three tools. **Every AI agent working on this codebase MUST use them and MUST NOT bypass them.**

### uv — package manager and task runner
```
Command              Purpose
uv sync              Install all deps (core + dev)
uv sync --all-extras Install all deps including optional backends
uv run pytest        Run tests
uv run <script>      Run any Python script within the venv
uv add <pkg>         Add a dependency
uv add --dev <pkg>   Add a dev dependency
uv add --optional <group> <pkg>  Add an optional dependency
uv lock              Regenerate lockfile after manual pyproject.toml edits
```

Rules:
- **MUST** use `uv` for all package management. Never use `pip`, `pip-tools`, `poetry`, or `pipenv`.
- **MUST** run all commands through `uv run` (e.g. `uv run pytest`, `uv run ruff check .`).
- **MUST NOT** activate a venv manually — `uv run` handles this.
- **MUST** keep `uv.lock` committed and up to date.

### ruff — linter and formatter
```
Command                    Purpose
uv run ruff check .        Lint the codebase
uv run ruff check --fix .  Auto-fix lint violations
uv run ruff format .       Format all Python files
uv run ruff format --check .  Check formatting (CI)
```

Rules:
- **MUST** run `ruff check .` before committing. Zero warnings allowed.
- **MUST** run `ruff format .` before committing. All files must be formatted.
- **MUST NOT** use `black`, `isort`, `flake8`, `pylint`, or `autopep8`. Ruff replaces all of them.
- **MUST NOT** add `# noqa` comments without a clear justification.
- Ruff configuration lives in `[tool.ruff]` in `pyproject.toml`. Do not create `.ruff.toml` or `ruff.toml`.

### ty — type checker
```
Command                 Purpose
uv run ty check .       Type-check the entire project
uv run ty check src/    Type-check a specific directory
```

Rules:
- **MUST** run `ty check .` before committing. Zero type errors allowed.
- **MUST NOT** add `# type: ignore` without a clear justification.
- **MUST** annotate all public function signatures with complete type hints.
- **MUST** use `| None` syntax (Python 3.10+), not `Optional[...]`.
- Ty configuration lives in `[tool.ty]` in `pyproject.toml`. Do not create `ty.toml`.

---

## Project conventions

### Code style
- Python >= 3.11 (see `requires-python` in `pyproject.toml`).
- All type hints required on public interfaces. Internal helpers may use inference where obvious.
- Docstrings: Google style for public API; concise for internal functions.
- Maximum line length: 100.
- Imports sorted: stdlib → third-party → project-local, each group separated by a blank line.
- Use `dataclasses` for plain data; `Pydantic` for config/validation/serialization boundaries.
- Use `ABC` + `@abstractmethod` for interfaces; use Protocol only when structural subtyping is truly needed.

### Package structure
```
src/llm_grammar_bench/     # All source code
tests/                     # All tests, mirroring src/ structure
configs/                   # YAML configuration files
```

- Package is `llm_grammar_bench`, importable as `import llm_grammar_bench`.
- Entry point: `main.py` at repo root (thin wrapper: `from llm_grammar_bench.cli import main; main()`).
- Scripts and console entry points are defined in `[project.scripts]` in `pyproject.toml`.

### Testing
- Framework: `pytest` with `pytest-cov` for coverage.
- Run: `uv run pytest` or `uv run pytest tests/test_module.py`.
- Tests mirror source structure: `src/llm_grammar_bench/backends/openai.py` → `tests/test_backends/test_openai.py`.
- **MUST** write tests for all new public interfaces.
- **MUST NOT** mock HTTP at the transport layer. Use `responses` or `pytest-httpx` for API backends.
- **MUST NOT** require GPU or network access in unit tests. Tests must pass in CI without API keys.
- Use `conftest.py` for shared fixtures; keep fixtures minimal.

### Dependencies
- Core dependencies are in `[project.dependencies]`. Keep this list minimal (no ML deps).
- Backend-specific deps are in `[project.optional-dependencies]` groups: `openai`, `anthropic`, `huggingface`, `openrouter`.
- Dev deps (ruff, ty, pytest, etc.) are in `[dependency-groups].dev`.

---

## Pre-commit checklist

Before marking any task as done, run:
```bash
uv run ruff check .        # Zero warnings
uv run ruff format --check .  # All formatted
uv run ty check .           # Zero type errors
uv run pytest               # All tests pass
```

If any command fails, the task is **NOT** done. Fix the issue and rerun.

## Coding etiquette

These rules govern how code is written, not just what it does. Violations are as serious as failing tests.

### Naming
- **MUST** use descriptive, pronounceable names. No single-letter variables except loop indices (`i`, `j`) and coordinates (`x`, `y`).
- **MUST NOT** abbreviate unless the abbreviation is universally understood in the domain (`id`, `url`, `json`, `http`, `api`, `cefr`).
- **MUST** use verbs for functions (`compute_scores`, `load_dataset`), nouns for classes (`MetricsRunner`, `BEA2019Dataset`), and booleans prefixed with `is_`/`has_`/`should_`.
- **MUST NOT** shadow built-in names (`id`, `type`, `list`, `dict`, `str`, `input`). Append a qualifier instead (`example_id`, `config_type`).

### Function design
- **MUST** keep functions small. If a function exceeds 30 lines, split it — or justify why in a comment.
- **MUST** follow single-responsibility: a function does one thing, stated in its name. If the name needs "and", split.
- **MUST NOT** produce side effects in functions that appear pure. A function named `load_dataset` must not modify global state or write files.
- **MUST** prefer early returns over deep nesting. Flatten conditionals with guard clauses.
- **MUST NOT** use mutable default arguments. Use `None` + internal assignment instead.

### Constants and configuration
- **MUST NOT** embed magic numbers in code. Name them as module-level constants: `MAX_RETRIES = 3`, not `for _ in range(3)`.
- **MUST NOT** hardcode paths, URLs, or credentials. They belong in config, env vars, or constants at the top of the module.
- **MUST** make default values explicit at the call site or in a config dataclass, never buried deep in a helper.

### Code cleanliness
- **MUST NOT** leave commented-out code. Delete it — git history exists for a reason.
- **MUST NOT** leave `# TODO` without a tracking reference (issue number or ticket ID). Untracked TODOs are lies.
- **MUST NOT** use `print()` for debugging. Use `logging` with appropriate levels (`logger.debug` for dev, `logger.info` for operational events).
- **MUST NOT** duplicate logic. If the same pattern appears twice, extract it. Three occurrences is a defect.
- **MUST** remove unused imports, variables, and dead code paths before marking work as done.

### Comments and documentation
- **MUST** write docstrings for all public modules, classes, and functions following Google style.
- **MUST** comment *why*, not *what*. The code already says what. Comments explain intent, tradeoffs, and non-obvious constraints.
- **MUST NOT** write redundant comments: `x += 1  # increment x` is noise. Delete it.
- **MUST** update docstrings when behavior changes. Outdated documentation is worse than no documentation.

### Error handling and resilience
- **MUST** handle errors at the appropriate level. Low-level functions raise; high-level functions handle and recover.
- **MUST** provide actionable error messages. `"API key not set. Set OPENAI_API_KEY in your environment."` — not `"KeyError"`.
- **MUST NOT** use `assert` for runtime validation. `assert` is for invariants that must never be false; use explicit `if`/`raise` for user-facing errors.
- **MUST** include context in exceptions: `raise BackendError(f"OpenAI request failed ({status}): {body}")` — not `raise BackendError("failed")`.
- **MUST NOT** catch `Exception` broadly. Be specific about which exception types you handle.
- **MUST** set timeouts on all network calls. Default: 60s per request.
- API backends **MUST** retry on 429 and 5xx with exponential backoff (max 3 attempts).

### Design principles
- **MUST** prefer composition over inheritance. Use ABCs for interfaces; use dependency injection for behavior.
- **MUST NOT** build abstraction for a single implementation. Abstract only when a second implementation exists or is imminent.
- **MUST** prefer immutable data. Dataclasses with `frozen=True` for value objects; copy-and-modify over in-place mutation.
- **MUST NOT** let one module grow unbounded. 300 lines is a warning; 500 lines is a refactoring requirement.
- **MUST** keep the public API surface small. If a symbol does not need to be imported by callers, prefix it with `_`.

### Logging
- **MUST** use the standard `logging` module. No `print()`, no custom log formatting unless required.
- **MUST** use appropriate levels: `DEBUG` for granular diagnostics, `INFO` for progress/benchmark events, `WARNING` for recoverable issues, `ERROR` for failures that affect results.
- **MUST** log at the boundary: API calls, file I/O, metric computation start/end.
- **MUST NOT** log API keys, tokens, or full request/response bodies at INFO level or above.

### Git practices
- **MUST** write commit messages in imperative mood: "Add retry logic to OpenAI backend", not "Added retry logic".
- **MUST** keep commits focused. One logical change per commit. Do not bundle unrelated refactors with feature work.
- **MUST NOT** commit generated files (`__pycache__`, `.egg-info`, `*.pyc`) — they are in `.gitignore`.
- **MUST NOT** commit secrets, API keys, or credentials. Use `.env.example` files with placeholder values.
---
