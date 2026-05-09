# Rust / PyO3 Design And Architecture Standards

These standards apply to Rust code and PyO3 bindings in `kaos-ml-core`.

## When To Use Rust

Use Rust for stable, measured bottlenecks or safety-sensitive parsing
where it materially improves performance, memory use, or reliability.
Do not add Rust only for novelty or for code that is still changing
rapidly at the API level.

Good Rust candidates:

- Tight CPU-bound loops.
- Graph, token, span, parser, matcher, and ranking kernels.
- Large batch transformations.
- Input validation where memory safety matters.
- Algorithms with clear inputs, outputs, and benchmarks.

Poor Rust candidates:

- Provider orchestration.
- Business logic with frequent product changes.
- Code whose main cost is network I/O.
- Thin wrappers around Python libraries.

## Layering

Use three layers:

| Layer | Responsibility |
|---|---|
| Rust core | Pure Rust data structures, algorithms, parsers, and errors. |
| PyO3 bindings | Minimal conversion between Python and Rust types. |
| Python wrapper | Typed public API, docs, dataclasses/Pydantic models, ergonomic errors. |

Rules:

- The Rust core should not depend on Python.
- PyO3 binding code should stay thin.
- Public users should normally import Python wrappers, not raw binding
  classes.
- Keep raw extension modules private by convention.

## Public API

- Expose stable Python names from `kaos_ml_core` wrapper modules.
- Keep Rust structs and Python classes named intentionally.
- Provide `.pyi` stubs or typed Python wrappers for public surfaces.
- Include `py.typed`.
- Document serialization, equality, ordering, hashing, and pickle
  behavior where applicable.

## Errors

- Define Rust error enums with clear variants.
- Map Rust errors to package-specific Python exceptions.
- Do not collapse every Rust failure into `ValueError`.
- Include safe context: input type, limit exceeded, unsupported format,
  or invalid structure.
- Exclude raw bytes, large payloads, secrets, internal paths, and
  provider responses from error strings.

## Memory And Ownership

- Prefer owned Rust data for long-lived structures.
- Avoid returning borrowed data tied to temporary Rust objects.
- Convert large byte buffers intentionally; avoid unnecessary copies
  where a safe zero-copy path exists.
- Keep Python object references out of Rust core types.
- Release the GIL for CPU-heavy Rust work when safe.
- Reacquire the GIL only for Python object interaction.

## Unicode, Bytes, And Offsets

- Be explicit about byte offsets versus character offsets.
- Public Python APIs should use Python string semantics unless a bytes
  API is clearly documented.
- Convert offsets at boundaries and test with non-ASCII inputs.
- Do not assume UTF-8 validity for arbitrary binary formats.

## Safety And Limits

- Avoid `unsafe` unless it has a narrow, documented reason and tests.
- Add hard iteration caps to iterative algorithms.
- Add input size, recursion, row, node, edge, page, and time limits
  where untrusted inputs can trigger expensive work.
- Fuzz parsers and binary/text decoders.
- Add regression tests for panics, overflows, invalid encodings, and
  malformed inputs.

## Build And Packaging

- Use maturin for Python packaging.
- Keep Cargo metadata aligned with Python package metadata.
- Use abi3 when the package policy requires stable Python ABI wheels.
- Verify wheel tags before release.
- Build wheels for the supported platform matrix.
- Do not ship local build artifacts, target directories, or generated
  shared libraries unless they are intentional package artifacts.

## Versioning

- Python package version is the user-facing version.
- Cargo prerelease strings and Python PEP 440 versions must be mapped
  consistently.
- `__version__` should reflect installed package metadata.
- Release tags should match the Python package version.

## Rust Quality Gates

Required before release:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
cargo deny check
uv run maturin build --release
```

Also run Python gates against the built extension:

```bash
uv run ruff format --check python tests
uv run ruff check python tests
uv run ty check python tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

## Benchmarks

- Keep a benchmark for every Rust path justified by performance.
- Compare against the prior Python implementation or a known baseline.
- Record input sizes and environment details.
- Treat large regressions as release blockers unless explicitly
  accepted.
