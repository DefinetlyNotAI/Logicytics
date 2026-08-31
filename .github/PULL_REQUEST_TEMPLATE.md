## Summary

<!-- Explain the problem, the focused solution, and the user-visible result. -->

## Contract impact

<!-- List affected collectors, modes/profiles, capabilities, settings, outputs, API, migration, or security boundaries. -->

## Verification

<!-- Include exact commands and results. Explain any skipped live Windows probe. -->

- [ ] Relevant focused tests pass.
- [ ] `python -m unittest discover -v` passes.
- [ ] `python -m compileall -q logicytics core tests` passes.
- [ ] `python -m logicytics preflight` reports no invalid core collector.
- [ ] `git diff --check` passes.
- [ ] Live Windows integration tests pass when the change touches host behavior.

## Review checklist

- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md) and searched for duplicate work.
- [ ] This pull request contains one coherent theme and conventional commits.
- [ ] Collector metadata, cancellation, isolation, and artifact registration remain valid.
- [ ] Configuration, output, migration, release, and feature-status docs are updated when affected.
- [ ] No generated evidence, credentials, private data, caches, or unrelated changes are included.
- [ ] I agree to the [Developer Certificate of Origin](../DCO.md) and repository license.

## Related issues

<!-- Use `Fixes #123` or `N/A`. -->
