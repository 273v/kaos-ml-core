# Repository Agent Instructions

## Scope

These instructions are the canonical repository-local guidance for coding
agents working in this repository. They apply to the whole tree unless a
more specific `AGENTS.md` appears in a subdirectory.

Keep changes focused on the requested task. Preserve unrelated user
changes, avoid generated artifacts in commits, and do not move public
release tags or force-push protected branches.

## Project Identity

- Distribution name: `kaos-ml-core`.
- Import package: `kaos_ml_core`.
- Python entry points: `kaos-ml` and `kaos-ml-serve`.
- Runtime: Python 3.13+ with a Rust extension built through PyO3 and
  maturin.
- Package role: classical machine-learning primitives over
  `kaos-content` AST and corpus abstractions, including corpus handling,
  clustering, labeling support, splits, training, metrics, thresholds,
  prediction, aggregation, and MCP tools.

## Setup

Use `uv` for Python environments, dependency resolution, and package
commands:

```bash
uv sync --group dev
uvx pre-commit install
```

Use `maturin` for the Python extension build. Keep Cargo metadata,
Python package metadata, the maturin module name, and wheel behavior
aligned with `pyproject.toml` and `Cargo.toml`.

## Local Checks

Before opening a PR, run the documented Python gate from
[CONTRIBUTING.md](CONTRIBUTING.md):

```bash
uv run ruff format --check python/kaos_ml_core tests
uv run ruff check python/kaos_ml_core tests
uv run ty check python/kaos_ml_core tests
uv run pytest tests/ -m "not live and not benchmark" --no-cov
```

For Rust/PyO3 changes, also run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
cargo deny check
uv run maturin build --release
```

Use `ty`, not mypy. Inline type suppressions must use the narrowest
practical `# ty: ignore[...]` form with a reason when the reason is not
obvious.

## Architecture Rules

Follow the detailed standards instead of duplicating them here:

- [Python design and architecture](docs/standards/python-design-and-architecture.md)
- [Rust/PyO3 design and architecture](docs/standards/rust-pyo3-design-and-architecture.md)
- [Code quality standards](docs/standards/code-quality-standards.md)
- [Tests, fixtures, fuzzing, and CI standards](docs/standards/tests-fixtures-ci.md)
- [Engineering process](docs/standards/engineering-process.md)

Keep public Python wrappers typed, ergonomic, and thin around lower-level
implementation details. Public users should normally import stable names
from `kaos_ml_core`, not raw private extension modules.

Keep stable, measured numeric and classical-ML hot paths in Rust where
Rust materially improves performance, memory use, or reliability. Keep
rapidly changing orchestration and integration behavior in typed Python.
PyO3 bindings should handle conversion and error translation, not domain
logic.

Preserve maturin wheel behavior, including the private extension module
shape and abi3 expectations. Translate Rust errors into package-specific
Python exceptions with safe, bounded context. Avoid `unsafe` unless a
narrow justification, review rationale, and tests are included.

When behavior touches document content, operate on `kaos-content` AST,
`Corpus`, `CorpusUnit`, `block_ref`, and tabular prediction abstractions
where relevant. Preserve stable mappings back to source documents.

Keep optional and large dependencies explicit through declared extras and
lazy imports. Do not make the base install depend on model downloads,
provider credentials, live services, or undeclared transitive packages.

## Testing

Test public behavior through real public entry points. For Rust/PyO3
work, test both the Rust side and the Python boundary that users call.

ML tests must stay deterministic: pin seeds where randomness is involved,
assert stable model, labeling, label-ordering, and class-ordering
behavior, and keep threshold, split, aggregation, and persistence
behavior reproducible. Use small, redistributable fixtures and mark live,
benchmark, and slow tests explicitly.

Bug fixes need regression tests. User-visible API, CLI, MCP schema,
package metadata, or release behavior changes need docs and changelog
consideration according to [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Never commit secrets, private keys, credentials, `.env` files, customer
data, privileged content, or unknown-license fixtures. Keep error
messages, logs, CLI output, JSON output, and test artifacts free of
secrets and private filesystem details.

Validate untrusted inputs at boundaries. Add limits for large corpora,
paths, archives, payloads, model artifacts, provider calls, and Rust/PyO3
boundaries where expensive or unsafe behavior is possible. Report
suspected vulnerabilities through [SECURITY.md](SECURITY.md), not public
issues.

## Commits, PRs, And Releases

Use focused conventional commits signed with the Developer Certificate of
Origin:

```bash
git commit -s -m "docs: update guidance"
```

Stage only files that belong to the requested change. Do not include
caches, wheels, shared libraries, `target/`, virtual environments,
coverage output, or generated files unless the task explicitly requires
them.

Before pushing, fetch `origin` and rebase on `origin/main` when local
`main` is behind. Push normally; do not force-push.

PRs should explain what changed, why it changed, how it was tested, and
whether public API, CLI behavior, MCP schemas, package metadata, fixtures,
or release artifacts are affected. Releases follow
[docs/standards/engineering-process.md](docs/standards/engineering-process.md)
and require clean formatting, linting, typing, tests, security checks,
built artifacts, strict metadata checks, and fresh install smoke tests.
