# Verification guide

Run from the repository root with the managed interpreter:

```powershell
python -m unittest discover -v
python -m compileall -q logicytics core tests
python -m logicytics preflight
git diff --check
```

For a documentation-only change, at minimum run the documentation tests, compile check, and diff check. For engine or collector changes, run the full suite and preflight. For Windows behavior, run the relevant integration tests on the target host; a non-Windows pass is not equivalent evidence.

The documentation tests verify the configuration example, guide links, collector specialty coverage, modes, output MIME types, and required entry points. The workflow itself should be reviewed as YAML and tested by a real Actions run after it is enabled.
