# v4 feature ownership and status

This register maps every section in [TODO.md](TODO.md) to its implementation
owner, current status, and primary evidence. Every checklist item inherits the
owner of its nearest heading; the checkbox beside that item is the authoritative
item-level status. `Wiki pending` means repository code and versioned docs are
complete but the separate GitHub wiki checkout still needs synchronization.

| TODO section | Owner | Status | Primary evidence |
| --- | --- | --- | --- |
| Core product redesign — mandatory v4.0 change | Architecture maintainers | Complete | `logicytics/contracts.py`, `planner.py`, `runtime.py`, `packaging.py` |
| Core functionality status | Runtime maintainers | Complete | `tests/test_core_functionality.py`, `tests/test_flow_matrix.py` |
| Core and plugin layout — canonical v4.0 model | Discovery maintainers | Complete | `core/`, `plugins/`, `logicytics/discovery.py` |
| Strict collector file, class, and specialty standards | Collector contract maintainers | Complete | `logicytics/contracts.py`, `discovery.py`, `tests/test_shipped_collectors.py` |
| File and folder rules | Discovery maintainers | Complete | `logicytics/discovery.py`, static preflight tests |
| Class and identity rules | Collector contract maintainers | Complete | `CollectorMetadata`, strict preflight tests |
| Specialty rules | Collector contract maintainers | Complete | `Specialty`, shipped collector catalog tests |
| Required class interface | Collector contract maintainers | Complete | `CoreCollector`, `PluginCollector`, lifecycle tests |
| Static and runtime preflight validation | Discovery maintainers | Complete | `logicytics/discovery.py`, `validation_worker.py` |
| Per-collector sandbox and failure isolation | Runtime maintainers | Complete | `logicytics/runtime.py`, cancellation/isolation tests |
| Replace the current architectural model | Architecture maintainers | Complete | v4 packages and `MIGRATION.md` |
| Define the v4.0 run model | Runtime and manifest maintainers | Complete | `runtime.py`, `manifest.py`, package integration tests |
| Separate responsibilities into layers | Architecture maintainers | Complete | CLI, planner, runtime, adapters, artifacts, packaging, maintenance modules |
| Make collection safe, bounded, and observable | Runtime and security maintainers | Complete | capability metadata, worker limits, progress/failure tests |
| Make concurrency a planned capability | Planner/runtime maintainers | Complete | resource scheduler and concurrency tests |
| Make configuration and profiles understandable | Configuration/planner maintainers | Complete | `CONFIGURATION.md`, `configuration.py`, profile tests |
| Improve the extension model | Extension maintainers | Complete | `MODS.md`, plugin/MOD discovery and isolation tests |
| Improve packaging and evidence integrity | Artifact/packaging maintainers | Complete | `OUTPUTS.md`, package/hash reproduction tests |
| Improve the public API and testability | API maintainers | Complete | `logicytics/api.py`, public API tests |
| Migration rules from the old design | Compatibility maintainers | Complete | `MIGRATION.md`, migration regression tests |
| v4.0 goals | Release maintainers | Complete | `V4_RELEASE.md` and full release gates |
| 1. Main engine and execution lifecycle | Runtime maintainers | Complete | CLI/planner/runtime integration tests |
| Execution modes | Mode/planner maintainers | Complete | `logicytics/modes.py`, machine-readable mode matrix tests |
| Post-run actions | Runtime maintainers | Complete | typed shutdown/reboot package-gating tests |
| 2. Command-line and interaction features | CLI/interaction maintainers | Complete | `cli.py`, `interaction.py`, usage/matching tests |
| 3. Core Logicytics engine/library API | API maintainers | Complete | public lazy API and integration tests |
| Command execution | Platform adapter maintainers | Complete | `ProcessAdapter`, command/MOD tests |
| File discovery | Discovery maintainers | Complete | `file_listing.py`, recursive discovery tests |
| Environment checks | Platform adapter maintainers | Complete | `environment.py`, adapter and live Windows tests |
| Configuration | Configuration maintainers | Complete | `CONFIGURATION.md`, parser/migration tests |
| Logging | Logging maintainers | Complete | `logging.py`, singleton/redaction/retention tests |
| Output and packaging | Artifact/packaging maintainers | Complete | `artifacts.py`, `packaging.py`, `OUTPUTS.md` |
| 4. Standard system-data collectors | Core collector maintainers | Complete | 66-collector preflight, lifecycle, output, and Windows tests |
| System, hardware, and operating-system inventory | System/hardware collector maintainers | Complete | `core/system/`, `core/hardware/` |
| Network inventory and activity | Network/packet collector maintainers | Complete | `core/network/`, `core/packet/` |
| Wireless, Bluetooth, and removable-device history | Wireless/device collector maintainers | Complete | `core/wireless/`, `core/bluetooth/`, `core/usb/` |
| Windows logs, registry, storage, and encryption | Windows evidence collector maintainers | Complete | `core/event_log/`, `registry/`, `storage/`, `encryption/` |
| Memory and process diagnostics | Process/memory collector maintainers | Complete | `core/process/`, `core/memory/` |
| User files and media | File evidence collector maintainers | Complete | `core/browser/`, `filesystem/`, `media/`, `ssh/` |
| Sysinternals integration | Diagnostics collector maintainers | Complete | `core/diagnostics/sysinternals_report.py`, `sysinternals.py` |
| 5. Maintenance and developer features | Maintenance maintainers | Complete | `maintenance.py`, CLI maintenance flow tests |
| Debugger (`--debug`) | Maintenance maintainers | Complete | debug diagnostics tests and `FLOW_MATRIX.md` |
| Update action (`--update`) | Maintenance maintainers | Complete | authenticated/read-only check and explicit apply tests |
| Developer action (`--dev`) | Maintenance maintainers | Complete | integrity/version/developer check tests |
| 6. MODS extension system | Extension maintainers | Complete | `MODS.md`, MOD runner and package tests |
| 7. Configuration and output contract | Configuration/artifact maintainers | Complete | `CONFIGURATION.md`, `OUTPUTS.md`, contract tests |
| 8. v4.0 delivery phases | Release maintainers | Wiki pending | phases 1-4 complete; phase 5 awaits separate wiki checkout |
| Phase 1 — Foundation | Architecture maintainers | Complete | contracts, configuration, logging, adapters, run workspace |
| Phase 2 — Engine and packaging | Runtime/packaging maintainers | Complete | scheduling, modes, ZIP/hash, MOD packaging tests |
| Phase 3 — Core collectors | Core collector maintainers | Complete | all 66 core collectors pass strict preflight |
| Phase 4 — Deep collection and visualization | Deep-collection maintainers | Complete | deep profile, packet graph, usage SVG tests |
| Phase 5 — Maintenance and release readiness | Release maintainers | Wiki pending | repository release gates complete; external wiki synchronization pending |
| 9. v4.0 completion checklist | Release maintainers | Wiki pending | repository checklist complete except the separate wiki update |

## Maintenance rule

Any new TODO heading must be added to this register in the same commit. Any item
whose owner, status, or evidence changes must update this table and the relevant
versioned contract document. The documentation contract test enforces exact
heading coverage so features cannot become ownerless.
