## Summary

<!-- One-paragraph description of what this PR does and why it's needed. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation only

## Checklist

- [ ] Commits are signed off (`git commit -s`) — DCO required
- [ ] Tests added/updated for any behavior change
- [ ] **Rust** — `cargo fmt --check`, `cargo clippy --no-default-features --all-targets -- -D warnings`, `cargo test --no-default-features --lib`
- [ ] **Python** — `uv run ruff format --check python/kaos_ml_core tests`, `uv run ruff check python/kaos_ml_core tests`, `uv run ty check python/kaos_ml_core tests`, `uv run pytest -m "not live and not network and not slow" tests/`
- [ ] **Build** — `uv run maturin develop --release` succeeds
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if user-visible

## Related issues

<!-- "Closes #123" or "Refs #123" -->
