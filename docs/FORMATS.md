# Reading generated values

Start with the run's `manifest.json`. It is the index of truth: use its collector IDs and artifact records to find files rather than assuming a filename exists.

## Common fields

An artifact has an ID, collector ID, POSIX-style run-relative path, name, MIME type, evidence kind (`raw` or `derived`), byte size, collection timestamp, transformations, and SHA-256. JSON values may contain `null` when Windows did not provide a value. Text and CSV should be read as UTF-8 where possible; preserve raw bytes when a file is marked binary.

## Format guidance

- JSON: parse as a mapping or array; retain unknown fields for forward compatibility.
- CSV: use a header-aware reader and do not split on commas manually.
- HTML: open locally and treat it as untrusted data; it may contain host-derived strings.
- XML: parse with a safe XML parser and avoid resolving external entities.
- ZIP: verify the adjacent SHA-256 sidecar before extraction; extract to a new directory and reject path traversal.
- Graphviz DOT: render only in a viewer you trust and treat labels as untrusted text.
- Plain text: preserve line breaks; it may be a command report rather than a key/value document.

## Using values in a script

The public API can load a validated `RunSnapshot` and read bounded artifacts. Prefer the API to globbing the output tree because it revalidates manifest structure and collector ownership.

```python
from pathlib import Path
from logicytics import open_artifact, query_run

snapshot = query_run(Path("."), "run-0123456789abcdef0123456789abcdef")
for item in snapshot.artifacts:
    print(item.collector_id, item.media_type, item.relative_path)

data = open_artifact(Path("."), snapshot.run_id, "artifact.0123456789abcdef0123456789abcdef")
with data as stream:
    first_bytes = stream.read(1024)
```

Keep the run ID and artifact ID from the manifest. Do not trust a path supplied by a user until it has been checked against the manifest and the expected output root.
