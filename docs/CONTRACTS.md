# Engine contracts

The contracts in `logicytics/contracts.py` are the stable vocabulary shared by the engine and collectors.

## Metadata

`CollectorMetadata` declares `id`, `name`, semantic `version`, `specialty`, description, author, supported platforms,
capabilities, sensitive-data categories, dependencies, default profiles, timeout, memory/output/artifact limits,
retries, minimum contract version, parallel safety, and resource class. Metadata is declarative: constructing it must
not collect data or write files.

IDs are `core.<specialty>.<name>`, `plugin.<name>` or `mod.<name>`. Labels are lowercase snake case. Capabilities and
output MIME types must be declared exactly.

## Lifecycle

The collector interface is:

```python
class ExampleCollector(PluginCollector):
    @classmethod
    def metadata(cls) -> CollectorMetadata:
        ...

    def validate(self, context: CollectorContext) -> ValidationResult:
        ...

    @staticmethod
    def prepare(context: CollectorContext) -> ValidationResult:
        ...

    def collect(self, context: CollectorContext) -> CollectorResult:
        ...

    @staticmethod
    def finalize(context: CollectorContext, result: CollectorResult) -> CollectorResult:
        ...

    @staticmethod
    def estimate(context: CollectorContext) -> CollectionEstimate:
        ...

    @classmethod
    def dependencies(cls) -> tuple[str, ...]:
        ...

    def cleanup(self, context: CollectorContext) -> None:
        ...
```

`validate` is side-effect-free and runs before authorization and collection. `prepare` is for collector-local setup.
`collect` performs bounded work. `finalize` returns the publishable result. `cleanup` releases collector resources; the
engine owns worker and filesystem cleanup.

## Context and artifacts

`CollectorContext` exposes `run_id`, `collector_id`, private `workspace`, private `temporary_directory`, `artifacts`,
`logger`, settings, and `is_cancelled`. Use `setting_int`, `setting_float`, and `setting_str` for safe configuration
reads. Use `report_progress` for meaningful counters and check `is_cancelled` inside loops.

`ArtifactWriter.register_file(source, media_type, evidence_kind, transformations)` is the only publication path. The
source must be inside the worker workspace. Register only complete, non-empty files with the declared MIME type. Never
hand the engine an arbitrary absolute destination.

`CollectorResult` contains a terminal status, non-empty summary, artifacts, errors, and scalar metrics. Return `partial`
when useful evidence exists but the collector could not complete fully; return `skipped` when a prerequisite is absent;
return `failed` for an execution error.

## Validation rules

No import-time subprocesses, network requests, registry writes, collection, or output writes. Keep imports within the
supported boundary. Do not use dynamic code loading to evade preflight. The static validator and isolated runtime probe
are part of the contract, not optional review steps.
