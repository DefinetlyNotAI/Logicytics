# Results and output layout

Each run is owned by a fingerprinted directory below `runtime.output_root` (normally `output/data`):

```text
output/data/
  run/<unique-prefix>/
    manifest.json
    artifacts/<collector_id_with_underscores>/...
    logs/...
    reports/...
  zip/<unique-prefix>.zip
  hashes/<unique-prefix>.zip.sha256
  logs/...
```

The prefix starts at eight SHA-256 characters and grows only to resolve a collision. A package contains the manifest, metadata, reports, logs, and evidence under stable `raw` or `derived` sections. A direct collector or `--no-package` run may have only the manifest-backed run folder.

## Manifest reading order

1. Read `status`, `run_id`, request, and plan.
2. Count collector records by status.
3. Read each failed or skipped record's summary, errors, and actionable failure details.
4. Follow `artifacts` entries to exact files.
5. Verify package and hash metadata before sharing or extracting a ZIP.

## Stable artifact rules

Artifacts are collector-owned, use POSIX separators, carry a MIME type and SHA-256, and cannot escape their owner directory. Typical MIME-to-suffix mappings are `application/json` → `.json`, `text/csv` → `.csv`, `text/html` → `.html`, `text/plain` → `.txt`, `application/xml` → `.xml`, `application/zip` → `.zip`, and `text/vnd.graphviz` → `.dot`.

Completed evidence, manifests, packages, hashes, reports, and run logs are retained together until explicit user removal. Application-log rotation is separate and controlled by `logging.retention_days`.
