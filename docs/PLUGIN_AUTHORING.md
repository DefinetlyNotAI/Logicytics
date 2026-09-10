# Plugin authoring

This guide assumes you can write basic Python. A plugin is a user-owned `PluginCollector` placed under `plugins/` and
enabled with `--plugins`. It is deliberately opt-in and is isolated like a core collector.

## Minimal plugin

```python
from logicytics.module.contracts import (
    CollectorMetadata, CollectorResult, CollectorStatus, CollectorContext,
    PluginCollector, ValidationResult, ArtifactWriter,
)

class ExamplePluginCollector(PluginCollector):
    @classmethod
    def metadata(cls) -> CollectorMetadata:
        return CollectorMetadata(
            id="plugin.example",
            name="Example plugin",
            version="1.0.0",
            specialty="integration",
            description="Writes a bounded example report.",
            author="Your name",
            capabilities=(),
            output_media_types=("text/plain",),
            default_profiles=("standard",),
            timeout_seconds=30,
            maximum_output_bytes=1024 * 1024,
            maximum_artifact_files=5,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        report = context.workspace / "report.txt"
        report.write_text("hello from the plugin\n", encoding="utf-8")
        artifact = context.artifacts.register_file(report, media_type="text/plain")
        return CollectorResult(CollectorStatus.SUCCEEDED, "Example report created", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        return None
```

The unused `ArtifactWriter` import in the conceptual example may be removed. Keep the actual module self-contained and
import only supported contracts and adapters.

## Metadata checklist

Use an ID beginning with `plugin.`; a lowercase name, semantic version, supported specialty, one-line description,
author, output MIME types, profiles, capabilities, sensible timeout and quotas. Declare network, sensitive data,
privilege, dependencies, retry policy, parallel safety, and resource class honestly. Metadata must be deterministic and
side-effect-free.

## Implementation checklist

1. Do not perform work while the module is imported.
2. Implement `metadata`, `validate`, `collect`, and `cleanup`; add `prepare`, `finalize`, `estimate`, or `dependencies`
   only when useful.
3. Write only inside `context.workspace` or the provided temporary directory.
4. Publish only through `register_file`.
5. Bound loops, files, bytes, subprocess time, and memory.
6. Check `context.is_cancelled` during long work and report progress counters.
7. Return typed results for expected failure and partial states.
8. Test on a machine without optional Windows facilities.

## Enable and test

```powershell
python -m logicytics preflight --plugins
python -m logicytics plan --mode standard --plugins
python -m logicytics run --mode standard --plugins --acknowledge-authorization
```

If the plugin is sensitive, do not put it in a routine profile. Use an exact `--include plugin.example` selection and
document the authorization decision.
