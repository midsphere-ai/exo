# Repository Guidelines

## Project Structure & Module Organization
This repository is a UV workspace monorepo. The root `pyproject.toml` defines shared tooling and includes all packages under `packages/` (for example, `orbiter-core`, `orbiter-models`, `orbiter-web`).

Python package layout follows:
- `packages/orbiter-<name>/src/orbiter/...` (or `src/orbiter_web/...` for web backend code)
- `packages/orbiter-<name>/tests/` for package tests
- `tests/integration/` for cross-package integration and marathon tests

Documentation lives in `docs/`, runnable samples in `examples/`, and static assets in `assets/`.

## Build, Test, and Development Commands
- `uv sync`: install all workspace packages and dev dependencies in editable mode.
- `uv run pytest`: run full test suite.
- `uv run pytest packages/orbiter-core/tests/`: run one package’s tests.
- `uv run ruff check packages/`: lint Python code.
- `uv run ruff format --check packages/`: formatting check.
- `uv run pyright packages/orbiter-core/`: type-check core package.

For Orbiter Web (`packages/orbiter-web`):
- `npm install`
- `npm run dev` (Astro frontend + FastAPI backend)
- `npm run build`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, max line length 100.
- Ruff enforces formatting/import order; use double quotes.
- Public functions/classes require type hints and Google-style docstrings.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep source files near ~200 lines when practical; move private logic to `_internal/`.

## Testing Guidelines
- Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`).
- Name tests `test_<what>_<scenario>`.
- Keep test filenames unique across packages to avoid import collisions.
- Use markers like `integration` and `marathon` intentionally.
- Do not call real provider APIs in tests; use mock providers/fixtures.

## Commit & Pull Request Guidelines
- Follow conventional-style commits seen in history: `feat:`, `fix:`, `docs:`, `merge:`.
- Keep commit subjects imperative and specific (example: `fix: handle empty tool arguments`).
- Open PRs against `main` with linked issue/context.
- Include a short summary of affected packages/paths.
- List verification commands you ran (`pytest`, `ruff`, `pyright`).
- Add screenshots for UI changes in `orbiter-web`.

## Security & Configuration Tips
Use environment variables for secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`). Never commit API keys, tokens, or generated `.env` secrets.
