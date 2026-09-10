# Evidence review and handoff

Treat every completed run directory and ZIP package as potentially confidential. The engine records evidence provenance and integrity metadata, but the operator still decides whether the scope was appropriate, whether the output is complete enough for the question, and who may receive it.

## Find the authoritative run record

Runs are stored below `runtime.output_root`, normally `output/data`. A completed run owns a fingerprinted directory containing `manifest.json`, collector artifacts, logs, and reports. Packaging, when enabled, writes a ZIP and a SHA-256 sidecar under sibling `zip` and `hashes` directories. The exact layout and retention distinction are documented in [Results](OUTPUTS.md).

The manifest is the authority for interpreting a run. Do not infer full success from a console message, a populated directory, or the existence of a ZIP.

## Review sequence

Use this order before analysis, export, or handoff:

1. Confirm `run_id`, overall `status`, request, and plan match the approved collection scope.
2. Count collector records by status. Read every `failed`, `partial`, `skipped`, and cancelled record before relying on an aggregate conclusion.
3. For each non-success record, capture the summary, structured error details, remediation, and retry-safe indication. Decide whether it changes the answer to the original question.
4. Follow each registered artifact entry to its path, MIME type, byte count, timestamp, evidence kind, and SHA-256. Use the manifest catalog instead of browsing arbitrary files in the workspace.
5. If a package was requested, verify the package and its hash metadata before transporting it. Retain the manifest with the package so its contents remain interpretable later.

The engine may keep useful evidence from a partial run. Preserve that context; do not discard a partial result merely because one independent collector failed. Equally, do not call a partial result complete when the failed collector was needed for the question.

## Handoff checklist

Before sharing, record the case or change reference, operator, date and time, host or asset identifier under the applicable policy, command or plan used, configuration identity, manifest path, package path, and hash value. State the overall status and each relevant limitation. Keep this record outside public issues and source control.

Share only the smallest approved artifact set. Raw evidence may contain account names, SIDs, hostnames, paths, addresses, browser data, saved Wi-Fi material, or private keys. Sanitize a diagnostic excerpt before using it in a ticket. Never attach an entire `output` tree, raw ZIP, credentials, or key material to a public issue.

## Retention and deletion

Completed run evidence, reports, manifests, packages, hashes, and run logs stay together until explicit user removal. `logging.retention_days` controls application-log rotation only; it does not silently expire run evidence. Follow the governing retention policy and confirm that the manifest, ZIP, sidecar, and required logs are no longer needed before removing a run.

For artifact type conventions, use [Formats](FORMATS.md). For privacy and scope requirements, use [Safety](SAFETY.md).
