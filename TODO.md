# Logicytics v4.0 Recreation TODO

This is the complete feature inventory and recreation plan for Logicytics v4.0.
The checklist describes what Logicytics does and what v4.0 must provide; it does
not describe the quality of the current implementation.

## Core product redesign — mandatory v4.0 change

The v4.0 recreation must not be a collection of scripts launched from a large
dispatcher. The core idea must change to a run-oriented evidence pipeline:

`request -> validated plan -> isolated collectors -> normalized artifacts -> manifest -> package`

Every feature in this file remains in scope, but each feature must fit that
pipeline. A collector is a producer of evidence, not an application that decides
where the whole product stores files, how the process exits, or how another
collector is launched.

### Core functionality status

- [x] v4 project metadata and standard-library-only engine foundation.
- [x] Typed collector contracts, metadata, statuses, run requests, validation, and
      artifact interfaces.
- [x] Typed local configuration with validated worker and output limits.
- [x] Strict discovery/preflight for `core/` and `plugins/`, including filename,
      class, inheritance, method, docstring, annotation, metadata, specialty, and
      ID validation.
- [x] Short-lived isolated metadata probe and invalid-plugin quarantine policy.
- [x] Deterministic profile/include/exclude planning with dependency ordering.
- [x] One worker process and private workspace per collector, with per-collector
      timeout containment and independent failure results.
- [x] Workspace-bound artifact registration, size limits, checksums, run manifests,
      ZIP creation, and package-level SHA-256 sidecars.
- [x] `python -m logicytics` commands for preflight, planning, and supervised runs.
- [x] Core integration tests for strict preflight, quarantine, isolated execution,
      artifact boundaries, manifests, and packaging.
- [x] Structured engine/collector JSONL logging, capability enforcement,
      cancellation propagation, Windows process-tree cleanup, and per-collector
      duration/progress/artifact accounting.
- [x] Add the first actual shipped collector: `core.system.system_info`, a bounded
      non-sensitive system inventory that establishes the strict collector pattern.
- [x] Add `core.process.running_processes`, a capability-gated, bounded,
      non-verbose Tasklist CSV collector that skips expected access denials without
      affecting unrelated collectors.
- [x] Add `core.network.network_identity`, a capability-gated hostname/address
      report that does not probe remote hosts.
- [x] Add `core.memory.memory_snapshot`, a capability-free Windows aggregate
      physical, virtual, and page-file memory report.
- [x] Add `core.storage.logical_drives`, a capability-free logical-drive type and
      capacity report that does not enumerate file contents.
- [x] Add `core.hardware.windows_features`, a deep-profile, subprocess-gated
      optional-Windows-feature inventory.
- [x] Add `core.event_log.system_events`, a deep-profile, subprocess-gated,
      bounded System event-log CSV export.
- [x] Add `core.network.arp_cache`, a deep-profile, subprocess-gated, read-only
      ARP-cache report that sends no network traffic.
- [x] Add `core.network.routing_table`, a deep-profile, subprocess-gated,
      read-only IPv4 and IPv6 routing-table report.
- [x] Add `core.system.system_details`, a deep-profile, subprocess-gated,
      complete `systeminfo` text report.
- [x] Add `core.system.bios_info`, a deep-profile, subprocess-gated BIOS CIM
      inventory that renders a portable HTML evidence table.
- [x] Add `core.system.operating_system`, a deep-profile, subprocess-gated CIM
      report with OS caption, version, service-pack, language, and path details.
- [x] Add `core.system.computer_system`, a deep-profile, subprocess-gated CIM
      report with computer model, manufacturer, and processor counts.
- [x] Add `core.storage.physical_disks`, a deep-profile, subprocess-gated CIM
      inventory of physical disk models, interface types, and capacities.
- [x] Add `core.system.session_snapshot`, a deep-profile, subprocess-gated report
      with Windows build, memory, current user/SID, language, host, time, and drive.
- [x] Add `core.process.detailed_processes`, a deep-profile, subprocess-gated,
      bounded verbose Tasklist CSV report.
- [x] Add `core.network.active_connections`, a deep-profile, subprocess-gated,
      bounded Netstat report with endpoints, protocol, state, and owning PID.
- [x] Add `core.network.adapter_statistics`, a deep-profile, subprocess-gated,
      per-interface byte, packet, error, and discard-counter inventory.
- [x] Add `core.network.network_interfaces`, a deep-profile, subprocess-gated
      IPv4 address, netmask, broadcast, link-state, speed, and duplex inventory.
- [x] Add `core.network.connection_processes`, a deep-profile, subprocess-gated
      Netstat and Tasklist correlation report for connection process names and PIDs.
- [x] Add `core.wireless.wifi_profiles`, a deep-profile, subprocess-gated,
      saved Wi-Fi profile-name inventory that never reads key material.
- [x] Add `core.event_log.application_events`, a deep-profile, subprocess-gated,
      bounded Application event-log CSV export.
- [x] Add `core.storage.mounted_volumes`, a deep-profile, subprocess-gated,
      mounted volume-GUID and mount-point mapping report.
- [x] Add `core.encryption.bitlocker_status`, a deep-profile, subprocess-gated,
      read-only `manage-bde -status` report.
- [x] Add `core.encryption.bitlocker_volumes`, a deep-profile, subprocess-gated
      `Get-BitLockerVolume` JSON report.
- [x] Add `core.system.system_diagnostics`, a deep-profile, subprocess-gated
      architecture, CPU, page-size, and boot-time report.
- [x] Add `core.event_log.security_events`, a deep-profile, subprocess-gated,
      bounded Security event-log CSV export.
- [x] Add `core.process.process_memory`, a deep-profile, subprocess-gated
      aggregate per-process working-set, private, and virtual memory report.
- [x] Add `core.filesystem.system_drive_tree`, a deep-profile, filesystem-read,
      bounded recursive system-drive tree with configurable depth and entry limits.
- [x] Add `core.filesystem.system_drive_listing`, a deep-profile, filesystem-read,
      bounded threaded recursive system-drive listing.
- [x] Add `core.network.bandwidth_sample`, a deep-profile, subprocess-gated,
      configurable local adapter bandwidth sampler.
- [x] Add `core.bluetooth.bluetooth_addresses`, a deep-profile, subprocess-gated,
      paired Bluetooth name and address-like PnP identifier report.
- [x] Add `core.system.installed_updates`, a deep-profile, subprocess-gated
      installed Windows hotfix inventory.
- [x] Add `core.registry.installed_applications`, a deep-profile, registry-read
      installed application inventory across 64-bit and WOW6432Node uninstall keys.
- [x] Add `core.system.windows_services`, a deep-profile, subprocess-gated Windows
      service inventory with state, startup, account, path, and process metadata.
- [x] Add `core.registry.startup_applications`, a deep-profile, registry-read
      inventory of standard user and machine Run/RunOnce startup entries.
- [x] Add `core.hardware.battery_status`, a deep-profile, subprocess-gated battery
      charge, capacity, and runtime inventory.
- [x] Add `core.network.dns_cache`, a deep-profile, subprocess-gated sensitive DNS
      resolver-cache export that sends no network traffic.
- [x] Add `core.network.firewall_profiles`, a deep-profile, subprocess-gated
      read-only Domain, Private, and Public firewall-profile inventory.
- [x] Add `core.system.environment_posture`, a deep-profile, subprocess-gated
      read-only administrator, UAC, and PowerShell execution-policy report.
- [x] Add `core.wireless.wifi_interfaces`, a deep-profile, subprocess-gated
      Wi-Fi interface report with normalized `Wi-Fi`/`WiFi` interface names.
- [x] Add `core.wireless.wifi_profile_keys`, an explicitly sensitive-data-gated
      deep collector that exports saved Wi-Fi profile XML with key material.
- [x] Add `core.registry.hklm_backup`, an explicitly sensitive-data-gated deep
      collector that exports the local HKLM hive as a `.reg` backup.
- [x] Add `core.ssh.ssh_backup`, an explicitly sensitive-data-gated deep collector
      that archives the current user's bounded `.ssh` keys and configuration.
- [x] Add `core.media.media_backup`, an explicitly sensitive-data-gated deep collector
      that copies bounded JPG, JPEG, PNG, and MP4 evidence from Pictures and Videos.
- [x] Add `core.browser.browser_data_backup`, an explicitly sensitive-data-gated deep
      collector for bounded Edge, Chrome, Firefox, Opera, and Opera GX profile evidence.
- [x] Add `core.bluetooth.bluetooth_history`, a timestamped deep snapshot whose
      retained run artifacts provide isolated Bluetooth history evidence.
- [x] Add `core.packet.packet_capture`, an explicitly packet-capture/elevation-gated
      IPv4 metadata collector with configurable count and timeout settings.
- [x] Add `core.filesystem.sensitive_file_inventory`, an explicitly sensitive-data-gated
      deep collector that finds supported sensitive-named files and copies matches concurrently.
- [x] Add `core.system.windows_system_data_backup`, an explicitly sensitive-data-gated
      deep collector that copies bounded Group Policy, event-log, and Defender support data.
- [x] Add `core.packet.connection_graph`, a deep subprocess-gated DOT export of
      source/destination connection edges with protocol labels.
- [x] Add `core.diagnostics.sysinternals_report`, a deep collector that reports
      Sysinternals archive/binary state and consolidates available supported tool output.
- [x] Add `core.process.memory_map`, a deep metadata-only memory-region collector
      with bounded readable addresses, sizes, permissions, mapped paths, and indexes.
- [x] Add `core.storage.volume_details`, a capability-free detailed logical-volume
      report with type, filesystem, label, free space, and capacity.
- [x] Add `core.network.network_adapters`, a subprocess-gated `ipconfig /all`
      adapter inventory.
- [x] Add `core.bluetooth.paired_devices`, a deep-profile, subprocess-gated,
      read-only PnP Bluetooth device inventory.
- [x] Add `core.usb.usb_storage_inventory`, a deep-profile, registry-read-gated
      USB storage inventory.
- [x] Add `core.system.installed_drivers`, a deep-profile, subprocess-gated
      detailed Windows driver inventory.
- [x] Add `core.system.group_policy`, a deep-profile, subprocess-gated local
      Group Policy Result summary.

## Core and plugin layout — canonical v4.0 model

The shipped product and user extensions must be physically and conceptually
separate:

```text
Logicytics/
├── core/                         # shipped, trusted-by-default collectors
│   ├── system/
│   │   └── system_info.py
│   ├── network/
│   │   └── network_info.py
│   └── ...
├── plugins/                      # user-added collectors and integrations
│   ├── example_plugin.py
│   └── ...
└── logicytics/                   # engine, contracts, planner, runtime
```

- [x] Make `core/` the only source of shipped collection functionality.
- [x] Automatically discover and validate every eligible collector in `core/` at
      startup; core collectors are included by the selected built-in profile unless
      a profile explicitly excludes them.
- [x] Make `plugins/` an opt-in user extension area. Discover plugins automatically
      during preflight, but only run valid plugins when the user selects the plugin
      profile/flag or explicitly includes them.
- [x] Permit plugins to implement any useful collection specialty, including new
      Windows data sources, reports, visualizations, exports, or integrations, as
      long as they obey the v4.0 collector contract.
- [x] Keep plugin output, logs, permissions, and failures separated from core
      output while still including both in the final run manifest when selected.
- [x] Never silently execute arbitrary `.py` files merely because they exist in
      `core/` or `plugins/`.
- [x] Ignore caches, tests, examples, private helpers, virtual environments, and
      files beginning with `_` during collector discovery.
- [x] Require an explicit `plugin_id` and detect duplicate IDs before any collector
      launches.
- [x] Decide and document whether an invalid core collector blocks the release/run
      and whether an invalid plugin blocks only that plugin or the whole requested
      plugin profile. The default policy should be: invalid core = launch refusal;
      invalid explicitly selected plugin = launch refusal; invalid unselected
      plugin = quarantine and report without executing it.

### Strict collector file, class, and specialty standards

Every executable collector must be a deliberately shaped Python module. A file
that fails any mandatory rule is not a collector and must not be launched.

#### File and folder rules

- [x] Require core collectors at `core/<specialty>/<collector_name>.py`.
- [x] Require plugins at `plugins/<plugin_name>.py` or, for multi-file plugins,
      `plugins/<plugin_name>/main.py` with all support files below that plugin's
      directory.
- [x] Require lowercase `snake_case.py` filenames containing only letters, digits,
      and underscores.
- [x] Require the filename to describe one specialty and one primary operation;
      reject vague names such as `misc.py`, `utils.py`, `stuff.py`, or `main.py`
      outside a plugin package.
- [x] Require one runnable collector per file. Shared helpers belong in a clearly
      marked support module and may not be discovered as collectors.
- [x] Require UTF-8 source, valid Python syntax, and a module docstring explaining
      the collector's purpose and data category.
- [x] Reject symlinks/junctions that resolve outside the project and reject plugin
      support files that escape their plugin directory.

#### Class and identity rules

- [x] Require exactly one public collector class in each runnable module.
- [x] Require the class name to be the filename converted to PascalCase followed by
      `Collector`; for example `network_info.py` must expose
      `NetworkInfoCollector`.
- [x] Require core classes to inherit from `CoreCollector`.
- [x] Require plugin classes to inherit from `PluginCollector`.
- [x] Reject a module if the required class is missing, misspelled, duplicated, or
      has a second public class that could be mistaken for the collector.
- [x] Require a stable, lowercase dotted ID matching its location and class, such as
      `core.network.network_info` or `plugin.example_plugin`.
- [ ] Require class metadata constants or an immutable metadata object for:
      `ID`, `NAME`, `VERSION`, `SPECIALTY`, `DESCRIPTION`, `AUTHOR`, and
      `SUPPORTED_PLATFORMS`.
- [ ] Require plugin metadata to additionally declare requested capabilities,
      privilege level, sensitive-data categories, network access, estimated cost,
      timeout, maximum output, and minimum contract version.
- [x] Require IDs, names, specialties, and versions to match their schemas and
      reject duplicates or unsupported contract versions before launch.

#### Specialty rules

- [ ] Require `SPECIALTY` to be exactly one registered value, not an arbitrary free
      text string. Initial values should cover system, hardware, process, memory,
      filesystem, network, packet, wireless, Bluetooth, USB, browser, registry,
      event-log, storage, encryption, media, SSH, diagnostics, reporting, and
      integration.
- [ ] Require each collector to have one primary specialty. Related secondary
      categories may be declared as metadata but may not change the collector's
      identity.
- [ ] Require the specialty to match the declared artifact types and the collector's
      documented responsibility.
- [ ] Reject a collector whose name, metadata, class, or outputs claim one specialty
      while its implementation declares or performs an unrelated primary job.
- [ ] Require a separate collector when a feature has a different permission model,
      timeout profile, sensitive-data classification, or output contract.

#### Required class interface

- [x] Require the following methods with exact names and type-checked signatures:
  - [x] `metadata() -> CollectorMetadata`.
  - [x] `validate(context: CollectorContext) -> ValidationResult`.
  - [x] `collect(context: CollectorContext) -> CollectorResult`.
  - [x] `cleanup(context: CollectorContext) -> None`.
- [x] Permit optional `estimate(context) -> CollectionEstimate` and
      `dependencies() -> tuple[str, ...]` methods only when they pass the contract
      validator.
- [x] Require `collect()` to write through the artifact service and return artifact
      references; direct package manipulation is forbidden.
- [x] Require collectors to report structured progress and never use undocumented
      console output as their API.
- [ ] Require all expected failures to become typed result/error values rather than
      process exits.
- [x] Require docstrings for the class and every public method.
- [x] Reject import-time collection, threads, subprocesses, network calls, file
      writes, registry writes, prompts, or `sys.exit()` calls.

#### Static and runtime preflight validation

- [x] Parse every discovered module with the Python AST before importing it.
- [ ] Validate path, filename, module docstring, imports, public names, class
      inheritance, class name, metadata, method signatures, type annotations, and
      forbidden top-level statements.
- [x] Import each candidate in a short-lived validation subprocess with collection
      disabled and with a restricted environment.
- [x] Instantiate the class only after static validation succeeds.
- [ ] Validate metadata, dependencies, supported platform, requested capabilities,
      output declarations, timeout, and size limits at runtime.
- [ ] Run a no-op contract probe against `validate()` and reject unexpected side
      effects, hangs, exceptions, or malformed return values.
- [ ] Produce a preflight report listing valid, quarantined, and invalid collectors,
      with exact file/line/rule failures.
- [x] Refuse to launch before collection if a core collector or explicitly selected
      plugin fails validation.
- [x] Never “best effort” execute a malformed collector because another collector
      succeeded.
- [ ] Cache validation only with a source hash, interpreter version, contract
      version, and configuration hash; invalidate the cache whenever any changes.

### Per-collector sandbox and failure isolation

“Sandboxed” means every collector is isolated as an independently supervised unit;
it does not mean that a collector automatically has permission to access anything
on the machine. Access must be declared, approved, and enforced as far as the
Windows platform allows.

- [x] Launch each collector in its own worker process rather than importing and
      running all collectors in the main process.
- [x] Give each collector a private working directory:
      `ACCESS/RUNS/<run-id>/collectors/<collector-id>/`.
- [x] Give each collector a private temporary directory, stdout/stderr capture,
      structured event channel, and artifact staging area.
- [x] Pass only a serializable `CollectorContext`/request payload into the worker;
      do not share mutable engine objects across collectors.
- [ ] Enforce declared timeout, memory/output limits, file-count limits, and
      cancellation through the supervisor.
- [ ] Terminate and clean up only the failed collector's process and workspace when
      it crashes, hangs, exceeds limits, or is cancelled.
- [ ] Mark that collector `failed`/`cancelled` with its traceback stored in the
      run manifest, while allowing independent collectors to finish.
- [ ] Prevent a collector from changing another collector's files, result state,
      logger configuration, environment, or execution plan.
- [ ] Restrict artifact registration to paths inside the collector workspace and
      copy/stream only approved artifacts into the run artifact store.
- [ ] Provide declared capability gates for filesystem reads, registry reads,
      subprocesses, network capture, raw packet access, browser-data access, and
      sensitive-file access.
- [ ] Require elevated privilege and explicit user approval for collectors that need
      it; do not grant administrator access to every plugin by default.
- [ ] Capture subprocess trees so child processes do not survive a failed collector.
- [ ] Use OS/process isolation as the default boundary; treat Python-level import
      restrictions as validation and defense-in-depth, not as a complete sandbox.
- [ ] Ensure a collector cannot call the main CLI, trigger reboot/shutdown, alter
      configuration, install packages, update the repository, or launch another
      collector through the public context API.
- [x] Provide a supervisor heartbeat and last-progress timestamp for every worker.
- [ ] Make the final package include the isolation result and failure reason for
      every selected collector.

### Replace the current architectural model

- [ ] Replace global action/sub-action state with an immutable `RunRequest` that
      contains the selected profile, explicit collector selections, concurrency,
      output policy, cancellation token, and post-run action.
- [ ] Replace hard-coded filename lists and mode-specific branches with a typed
      collector registry. Each collector declares its ID, category, supported
      platforms, required privileges, estimated cost, dependencies, output types,
      default profiles, and whether it is safe to parallelize.
- [ ] Replace scripts writing directly into the repository with a `RunContext` that
      gives each collector a run ID, isolated workspace, logger, configuration
      snapshot, artifact writer, and cancellation handle.
- [ ] Replace implicit current-working-directory behavior with explicit absolute
      paths owned by the run context.
- [ ] Replace `print`, `exit`, and process-level shutdowns inside collectors with
      typed results and typed errors returned to the orchestrator.
- [ ] Replace one giant execution-list algorithm with a planner that resolves
      collector dependencies, removes duplicates, validates conflicts, and emits a
      reproducible execution plan before any collection begins.
- [ ] Replace mode names that imply implementation details with user-facing
      collection profiles. Preserve compatibility aliases for existing flags, but
      make profiles the canonical internal concept.
- [ ] Replace a single shared mutable logger with run-scoped structured logging
      that can still be rendered to the existing console/log-file formats.
- [ ] Replace ad hoc output filenames with an artifact catalog containing stable
      artifact IDs, human-readable names, MIME/type information, byte counts,
      timestamps, producer collector, and status.
- [ ] Replace “zip whatever happens to be in CODE” with packaging the run's
      artifact catalog and manifest only.
- [ ] Replace cleanup based on deleting a discovered file list with ownership-based
      cleanup that can remove only the current run's temporary workspace.

### Define the v4.0 run model

- [ ] Create a unique run directory before collection starts:
      `ACCESS/RUNS/<run-id>/`.
- [ ] Store raw collector output under a controlled artifact tree rather than in
      the source tree or project root.
- [ ] Create a run manifest at the beginning, update it as collectors start and
      finish, and finalize it even after cancellation or partial failure.
- [ ] Record requested profile, resolved collector plan, Logicytics version,
      configuration snapshot with secrets removed, host metadata, privilege state,
      start/end times, cancellation state, errors, skipped items, and artifact
      checksums.
- [ ] Give every collector an explicit lifecycle: validate, prepare, collect,
      finalize, report result.
- [ ] Make collector results explicit: `succeeded`, `partial`, `skipped`,
      `cancelled`, or `failed`.
- [ ] Allow partial runs to be packaged and clearly labeled instead of appearing
      successful or being silently discarded.
- [ ] Make reruns selectable by collector ID and keep rerun artifacts separate from
      the original run.
- [ ] Make the run plan deterministic: the same request and configuration produce
      the same collector set and dependency order.

### Separate responsibilities into layers

- [ ] **CLI layer:** parse arguments, show help, request confirmation, and render
      results; do not collect data or manipulate package files directly.
- [ ] **Application layer:** create the request, resolve the profile, plan the run,
      coordinate collectors, and decide the final run status.
- [ ] **Collector layer:** perform one focused collection task through injected
      services; do not import the main application or depend on global state.
- [ ] **Platform layer:** centralize PowerShell, WMIC/WMI, registry, filesystem,
      process, network, and privilege access behind testable adapters.
- [ ] **Artifact layer:** validate paths, stream output, normalize metadata, compute
      checksums, enforce size limits, and register artifacts.
- [ ] **Packaging layer:** consume the finalized artifact catalog and create ZIP,
      hash, manifest, and summary outputs.
- [ ] **Maintenance layer:** keep update, debug, developer checks, and usage
      analytics from being coupled to normal collection.

### Make collection safe, bounded, and observable

- [ ] Require an explicit consent/authorization acknowledgement before collection,
      with a clear summary of the selected categories and sensitive outputs.
- [ ] Mark collectors that may copy credentials, private keys, personal files,
      browser data, or packet contents as sensitive and make them opt-in for the
      default profile.
- [ ] Add per-collector timeouts, retry policies, maximum output size, maximum file
      size, maximum file count, and maximum total run size.
- [ ] Add cancellation checks inside long filesystem walks, packet capture,
      memory collection, event-log queries, and copy loops.
- [ ] Prevent path traversal and symlink/junction escapes from leaving the run
      workspace or copying unintended locations.
- [ ] Avoid loading entire command output, files, memory maps, or packet sets into
      memory when streaming is possible.
- [ ] Redact secrets from logs and from diagnostic metadata even when the artifact
      itself is intentionally collected.
- [ ] Never place credentials or raw sensitive evidence in console logs, exception
      messages, temporary filenames, or update requests.
- [ ] Record resource usage and progress per collector: files scanned, files
      copied, bytes written, packets observed, events processed, and elapsed time.
- [ ] Make failures actionable by recording the collector ID, operation, platform
      error, remediation hint, and whether a retry is safe.

### Make concurrency a planned capability

- [ ] Use a scheduler with bounded workers instead of creating independent thread
      pools inside collectors.
- [ ] Declare dependencies and resource classes so disk-heavy, network-heavy,
      registry-sensitive, and interactive collectors can be coordinated.
- [ ] Prevent concurrent collectors from writing the same artifact path.
- [ ] Use thread-safe artifact registration and structured event reporting.
- [x] Preserve deterministic manifest ordering even when collectors finish out of
      order.
- [ ] Define which collectors may run in parallel and which must run serially.
- [ ] Provide a sequential mode for debugging and a bounded parallel mode for
      normal use; do not make parallelism a hidden behavior of a particular flag.

### Make configuration and profiles understandable

- [ ] Replace scattered string lookups with a validated typed configuration object.
- [ ] Fail at plan-validation time for invalid values, rather than during a
      collector's execution.
- [ ] Separate product settings, profile settings, collector settings, and runtime
      overrides.
- [ ] Provide named profiles such as `minimal`, `standard`, `deep`, and `offline`
      with documented collector membership.
- [ ] Allow explicit include/exclude collector selections to override a profile.
- [ ] Version the configuration schema and migrate older configuration files.
- [ ] Keep remote configuration optional, authenticated/validated, and out of the
      critical local collection path.
- [ ] Never let a remote manifest silently add a collector to a user's run.

### Improve the extension model

- [ ] Define a stable `Collector` interface and version it.
- [ ] Require every MODS collector to declare metadata rather than being discovered
      only by extension.
- [ ] Run mods in a sandboxed/isolated run workspace with declared inputs and
      outputs.
- [ ] Give mods the same artifact writer, cancellation, timeout, logging, and
      result contract as built-in collectors.
- [ ] Prevent a mod from changing global configuration, replacing built-in
      collectors, or writing outside its permitted workspace without explicit
      permission.
- [ ] Preserve compatibility with existing script mods through an adapter layer,
      while making the typed collector API the v4.0 contract.

### Improve packaging and evidence integrity

- [x] Package only finalized artifacts referenced by the manifest.
- [ ] Stream ZIP creation and hash calculation to support large runs.
- [x] Add per-artifact SHA-256 values and a package-level SHA-256 value.
- [x] Include a machine-readable manifest and a human-readable summary report.
- [x] Include collector status, skipped/failed reasons, and collection timestamps
      in the summary.
- [x] Preserve artifact provenance: collector ID, source category, collection time,
      and transformation steps.
- [x] Verify package contents against the manifest before marking the run complete.
- [x] Use atomic writes and temporary package names so interrupted packaging cannot
      look like a finished result.
- [ ] Keep raw evidence, derived reports, logs, hashes, and metadata in separate
      package sections.

### Improve the public API and testability

- [ ] Make the package importable without starting a collection, opening files, or
      requiring Windows-only APIs at import time.
- [ ] Replace singleton-heavy APIs with dependency injection where state matters.
- [ ] Keep a small stable public API: configuration, planning, running, querying
      run status, and reading artifacts.
- [ ] Keep platform adapters mockable so collectors can be tested on non-Windows
      development machines.
- [ ] Add contract tests for every collector's metadata, lifecycle, result states,
      artifact declarations, and error behavior.
- [ ] Add planner tests for profiles, explicit selections, dependencies, conflicts,
      duplicate collectors, and invalid configurations.
- [ ] Add run tests for sequential, bounded-parallel, cancellation, timeout,
      partial-failure, rerun, and cleanup behavior.
- [x] Add packaging tests proving that unregistered files cannot enter a package
      and that manifest hashes match the final bytes.
- [ ] Add golden-output tests for stable text, CSV, HTML, graph, and manifest
      formats, allowing platform-specific values to be normalized.
- [ ] Add end-to-end Windows tests for permissions, PowerShell, registry, WMI,
      event logs, BitLocker, networking, and Sysinternals where available.

### Migration rules from the old design

- [ ] Keep existing command-line flags as compatibility aliases during migration.
- [ ] Map every legacy flag to a named v4.0 profile or explicit collector set.
- [ ] Wrap legacy scripts in compatibility collectors before rewriting their data
      gathering logic.
- [ ] Move one collector at a time from direct filesystem writes to artifact APIs.
- [ ] Keep legacy output names as aliases or compatibility copies only where users
      rely on them; make the run artifact tree canonical.
- [ ] Remove global mutable state after all collectors use `RunContext`.
- [ ] Remove direct `exit()` calls, current-directory assumptions, nested worker
      pools, and package-wide cleanup from collectors.
- [ ] Remove compatibility shims only after the v4.0 API and migration documentation
      cover their replacement.

## v4.0 goals

- [ ] Recreate Logicytics as a Windows system-data collection and packaging tool.
- [ ] Preserve every supported collection category, execution mode, output type,
      configuration option, and extension point listed below.
- [ ] Keep collection modules independently runnable as well as runnable through
      the main orchestrator.
- [ ] Make collection results predictable, timestamped, inspectable, and easy to
      verify after export.
- [ ] Keep administrator checks, permission-aware collection, cancellation, and
      failure reporting as first-class behavior.
- [ ] Keep the default run reasonably quick while retaining a deeper, slower mode
      for exhaustive collection.
- [ ] Keep the project Windows-focused, including PowerShell, registry, WMI,
      WMIC, Windows event logs, BitLocker, and Sysinternals integrations.

## 1. Main engine and execution lifecycle

- [ ] Provide a single main entry point that:
  - [ ] Parses command-line flags and validates their combinations.
  - [ ] Handles special actions before collection starts.
  - [ ] Checks privileges and relevant Windows prerequisites.
  - [ ] Extracts Sysinternals tools when permitted by configuration.
  - [ ] Builds the execution list for the selected mode.
  - [ ] Runs the selected Python, PowerShell, executable, and batch collectors.
  - [ ] Captures collector output and converts script messages into Logicytics log
        levels where applicable.
  - [ ] Continues collecting when an individual collector fails, while recording
        the failure.
  - [ ] Packages generated evidence after collection.
  - [ ] Produces a SHA-256 hash beside every package.
  - [ ] Performs the requested shutdown or reboot action at the end.
  - [ ] Handles Ctrl+C gracefully and attempts final packaging/cleanup.
  - [ ] Offers a final exit prompt for interactive use.
- [ ] Discover runnable files recursively from the CODE directory.
- [ ] Support `.py`, `.ps1`, `.exe`, and `.bat` collector types.
- [ ] Exclude engine/library directories and underscore-prefixed helper files from
      automatic collector discovery.
- [ ] Support a configurable worker limit for parallel collection.
- [ ] Provide per-collector success, failure, and duration information.

### Execution modes

- [x] `--default`: run the standard set of collectors sequentially.
- [x] `--threaded`: run the standard set concurrently.
- [x] `--minimal`: run the quick/basic collection set.
- [x] `--depth`: run the standard set concurrently plus the slow, large-data
      collectors.
- [ ] `--nopy`: run the non-Python collectors for systems without Python.
- [ ] `--modded`: run the normal collection set plus all supported files in MODS.
- [x] `--performance-check`: run collectors sequentially, measure each duration,
      and write a performance summary table.
- [ ] Show help when no collection action is selected.
- [ ] Reject multiple mutually exclusive collection actions.

### Post-run actions

- [ ] `--reboot`: schedule a system reboot after packaging.
- [ ] `--shutdown`: schedule a system shutdown after packaging.
- [ ] Ensure post-run actions do not run prematurely during performance analysis.

## 2. Command-line and interaction features

- [ ] Provide descriptive help text for every flag.
- [ ] Support the current action flags: `default`, `threaded`, `modded`, `depth`,
      `nopy`, `minimal`, `performance-check`, and `usage`.
- [ ] Support side actions: `debug`, `update`, and `dev`.
- [ ] Provide action/sub-action exclusivity validation and clear invalid-combination
      errors.
- [ ] Provide colored console output for status, warnings, errors, and prompts.
- [ ] Provide a semantic flag-matching feature that can map natural-language user
      input to the closest command-line flag.
- [ ] Match input against both flag names and their descriptions.
- [ ] Use a configurable similarity threshold.
- [ ] Fall back to historical flag suggestions when direct matching is weak.
- [ ] Persist optional compressed interaction history locally only when enabled.
- [ ] Record matched flag, input, accuracy, timestamp, and device name in history.
- [ ] Track per-flag usage counts.
- [ ] Provide `--usage` statistics including total interactions, average accuracy,
      common device/input values, and per-flag frequency.
- [ ] Generate a flag-usage bar graph in the ACCESS/DATA area.
- [ ] Allow model debug/progress output to be enabled independently from normal
      Logicytics debug logging.

## 3. Core Logicytics engine/library API

### Command execution

- [x] Expose a reusable command runner that returns command stdout.
- [ ] Execute Python collectors through the configured Python runtime.
- [ ] Execute PowerShell collectors after unblocking them when necessary.
- [ ] Execute other supported script types through PowerShell where applicable.
- [x] Parse `LEVEL: message` output from non-Python scripts into structured logs.

### File discovery

- [x] Provide recursive file listing with extension filters.
- [x] Support appending to an existing file list.
- [x] Support excluded files, extensions, and directories.
- [x] Normalize relative paths for execution and integrity comparisons.

### Environment checks

- [x] Check whether the process has administrator privileges.
- [x] Check the PowerShell execution policy.
- [x] Check whether UAC is enabled.
- [x] Detect and extract the bundled Sysinternals archive unless a local ignore
      marker opts out.
- [x] Report missing, archived, or extracted Sysinternals tools without stopping
      unrelated collection.

### Configuration

- [ ] Load `config.ini` from the project CODE directory.
- [ ] Expose debug level, version, current file manifest, log retention, worker
      count, preference persistence, and all collector settings.
- [ ] Support remote configuration retrieval for update/integrity checks.
- [ ] Keep configuration parsing available to every collector through the package.

### Logging

- [ ] Provide a singleton logger shared by the engine and collectors.
- [ ] Log to console with configurable colors and levels.
- [ ] Log to ACCESS/LOGS/Logicytics.log.
- [ ] Support DEBUG, INFO, WARNING, ERROR, CRITICAL, INTERNAL, and EXCEPTION levels.
- [ ] Include timestamps and structured, readable log rows.
- [ ] Support log truncation controls and optional deletion of the previous log.
- [ ] Provide raw logging, newline separators, typed message dispatch, and batched
      execution-message parsing.
- [x] Provide a function decorator that records function execution and timing.
- [x] Provide an exception helper that logs and raises a requested exception type.
- [x] Provide a deprecation decorator with removal version, reason, and optional
      stack trace.
- [ ] Preserve a dedicated DEBUG log and diagnostic context for `--debug`.

### Output and packaging

- [ ] Create ACCESS/LOGS, ACCESS/LOGS/DEBUG, ACCESS/LOGS/PERFORMANCE,
      ACCESS/DATA/Zip, and ACCESS/DATA/Hashes automatically.
- [ ] Collect generated files without including source code, executables, model
      files, configuration secrets, caches, or library internals in the evidence
      package.
- [ ] Include generated files from CODE in the package.
- [ ] Include MODS output in a separate MODS package during modded runs.
- [ ] Preserve nested directory structure inside packages.
- [ ] Name packages using the action and timestamp.
- [ ] Generate a SHA-256 digest for each package and move both artifacts into the
      ACCESS/DATA output directories.
- [ ] Provide a file-opening helper for viewing generated artifacts.
- [ ] Implement the v4.0 temporary-workspace lifecycle: collectors write into a
      run-specific temporary directory, the packager consumes that directory, and
      cleanup occurs after successful or interrupted runs.

## 4. Standard system-data collectors

Every collector below must remain independently runnable and available to the
appropriate default/deep/non-Python mode.

### System, hardware, and operating-system inventory

- [x] Run `systeminfo` and save the complete output.
- [x] Query BIOS manufacturer, name, and version through modern Windows CIM data and
      provide an HTML table representation.
- [x] Query operating-system caption, service-pack information, and related details
      through modern Windows CIM data.
- [x] Query computer model, manufacturer, and processor count.
- [x] Query disk-drive model and size.
- [x] Collect Windows build, physical/virtual memory, current user, SID, language
      IDs, computer name, date/time, and system drive.
- [x] Enumerate Windows optional features and their enabled/disabled state.
- [x] Collect local network adapter configuration through a bounded `ipconfig /all`
      report; structured adapter fields remain a later enhancement.
- [x] Collect installed/visible device-driver information with `driverquery /v`.
- [x] Collect group-policy result information with `GPResult /r`.
- [x] Export the running process/task list as CSV, including verbose details.
- [x] Generate a recursive file-system tree for the system drive with explicit bounds.
- [x] Generate a recursive directory listing of the system drive using the deep,
      threaded collector with explicit bounds.

### Network inventory and activity

- [x] Collect per-interface network I/O counters: bytes, packets, errors, and
      dropped traffic.
- [x] Collect active network connections, endpoints, protocol types, and status.
- [x] Collect network interface addresses, netmasks, and broadcast addresses.
- [x] Collect interface speed, duplex, and link-up state.
- [x] Save bounded `ipconfig /all` output.
- [x] Associate active connections with process names and PIDs.
- [x] Resolve and save the hostname and local IP addresses without probing remote hosts.
- [x] Measure average and peak upload/download bandwidth over configurable sample
      counts and intervals without generating network traffic.
- [x] Correct the Wi-Fi/WiFi interface-name variation automatically.
- [x] Sniff a configurable number of packets on a configurable interface with a
      timeout and retry window.
- [x] Record source/destination IP, protocol, source/destination ports, and packet
      metadata for IP/TCP/UDP/ICMP traffic.
- [x] Export packet observations to CSV.
- [x] Build and save a source/destination network graph with protocol edge labels.
- [ ] Clean up packet-sniffer graph state and plotting resources.

### Wireless, Bluetooth, and removable-device history

- [x] Enumerate saved Wi-Fi profiles through `netsh` without requesting key material.
- [x] Retrieve each saved profile's key content when available.
- [x] Save profile names and associated credentials to a report.
- [x] Query paired/available Bluetooth devices through PowerShell PnP data.
- [x] Save Bluetooth friendly name/instance ID, class, status, problem code, and
      presence state; manufacturer/description enrichment remains a later addition.
- [x] Enumerate paired Bluetooth devices with names and extracted address-like PnP identifiers.
- [x] Append timestamped Bluetooth collection sections to a historical report.
- [x] Enumerate USB storage device class, instance, friendly name, and last-write
      timestamps from the Windows registry.
- [x] Record USB device class, device ID, friendly name, and registry last-write
      time.

### Windows logs, registry, storage, and encryption

- [x] Export a bounded sample of up to 1,000 Windows System event-log entries to CSV;
      full log backup remains a later configurable deep collection.
- [x] Read Security, Application, and System event logs through the modern Windows event API.
- [x] Record event category, generated time, source, event ID, type, and message data
      in separate bounded CSV reports.
- [ ] Run the event-log parsers concurrently and write separate reports.
- [x] Export the HKLM registry hive to a `.reg` backup.
- [x] Enumerate logical volumes with drive type, filesystem, free space, size, and
      volume name.
- [x] Enumerate mounted volume GUIDs with `mountvol`.
- [x] Report BitLocker status for detected drive letters using `manage-bde`.
- [x] Report `Get-BitLockerVolume` output when PowerShell is available.
- [x] Include collection timestamp, user, admin state, hostname, and platform in
      the encrypted-volume report.

### Memory and process diagnostics

- [x] Capture RAM and page-file totals, used/available values, and utilization percent.
- [x] Capture architecture, OS, machine, processor, page size, CPU count/frequency,
      and boot time.
- [x] Create a process memory-map report containing readable regions, addresses,
      sizes, RSS, permissions, mapped path, and region index.
- [x] Enforce an optional output-size limit and disk-space safety margin.
- [x] Record truncation when the configured memory-report limit is exceeded.
- [x] Store memory diagnostics in a configurable dump directory.

### User files and media

- [x] Search the system drive for files whose names contain sensitive-data keywords
      such as password, secret, code, login, API, key, token, auth, credentials,
      private, certificate, SSH, PGP, or wallet.
- [x] Restrict sensitive-file mining to the supported document, data, archive,
      database, configuration, log, and text extensions.
- [x] Copy matching files into a dedicated report directory.
- [x] Use concurrent search/copy operations and skip files over the configured
      per-file size ceiling.
- [x] Back up the current user's Pictures and Videos directories.
- [x] Recursively collect supported image/video formats, including JPG, JPEG, PNG,
      and MP4.
- [x] Copy media with timestamped destination names while preserving file metadata.
- [x] Back up the current user's `.ssh` directory, including keys and configuration,
      into a dedicated SSH backup directory.
- [x] Copy browser-related data for Edge, Chrome, Firefox, Opera, and Opera GX.
- [x] Copy the configured Windows security, group-policy, and event-log data paths
      into labeled browser/system-data directories.

### Sysinternals integration

- [x] Support the bundled Sysinternals Suite archive and optional ignore marker.
- [x] Run available `psfile`, `PsGetsid`, `PsInfo`, `pslist`, `PsLoggedon`, and
      `psloglist` tools.
- [x] Append each tool's stdout and relevant stderr to a consolidated report.
- [x] Report absent binaries, archived-only state, extracted state, and execution
      errors.

## 5. Maintenance and developer features

### Debugger (`--debug`)

- [ ] Fetch the remote configuration manifest.
- [ ] Compare local and remote versions, including snapshot-version handling.
- [ ] Compare the local file tree to the configured required-file manifest.
- [ ] Report missing and extra files.
- [ ] Check Sysinternals archive/binary state.
- [x] Report admin/UAC state and PowerShell execution policy.
- [x] Report Python, psutil, executable, prefix, virtual-environment, CPU, and
      logging context.
- [ ] Report whether the running Python version is recommended, supported, or
      incompatible.
- [x] Write diagnostic output to a dedicated debug log.

### Update action (`--update`)

- [x] Verify that Git is available.
- [x] Verify that the project is a Git checkout.
- [x] Pull the configured upstream repository when the user explicitly requests it.
- [x] Report update success, failure, and repository-state errors.
- [ ] Support launching a selected action in a new command window when required by
      the update/special-action workflow.

### Developer action (`--dev`)

- [ ] Present contribution and repository-organization checks interactively.
- [ ] Check naming conventions, CODE placement, documentation/docstrings, and
      one-main-feature-per-file organization.
- [ ] Compare current files against the configured manifest.
- [ ] Display added, removed, and unchanged files with colored status markers.
- [ ] Update the manifest in `config.ini` after confirmation.
- [ ] Prompt for and validate the next semantic version.
- [ ] Keep excluded history/cache/Sysinternals files out of manifest comparisons.

## 6. MODS extension system

- [ ] Treat MODS as an opt-in extension directory.
- [ ] Discover `.py`, `.exe`, `.ps1`, and `.bat` mods.
- [ ] Run mods after the normal Logicytics collectors.
- [ ] Allow users to add custom collectors without changing the engine.
- [ ] Include mod-generated data in a separately named package.
- [ ] Document the mod contract, supported file types, execution order, output
      location, logging convention, and failure behavior.
- [ ] Keep underscore-prefixed helper files out of automatic mod execution.

## 7. Configuration and output contract

- [ ] Recreate general settings for debug logging, old-log deletion, worker count,
      and preference-history persistence.
- [ ] Recreate flag-matching settings for model name, minimum accuracy, and model
      debug output.
- [ ] Recreate memory-dump settings for file-size limit, safety margin, and dump
      directory.
- [x] Recreate network sampling settings for count and interval.
- [x] Recreate packet-sniffer settings for interface, packet count, timeout, and
      retry duration.
- [x] Version the configuration schema and provide migration/default handling.
- [ ] Define stable output names and formats for every collector.
- [ ] Keep logs, hashes, packages, graphs, CSVs, HTML, text, and copied evidence
      discoverable under one run-specific output tree.
- [x] Include collection metadata: version, action, start/end times, host, user,
      privilege state, enabled collectors, skipped collectors, and failures.

## 8. v4.0 delivery phases

### Phase 1 — Foundation

- [ ] Define the v4.0 package layout and run manifest format.
- [ ] Implement configuration loading, structured logging, file discovery, command
      execution, privilege checks, cancellation, and error aggregation.
- [ ] Implement the temporary run directory and cleanup lifecycle.
- [ ] Rename ACCESS folder to output/ and LOGS/ to logs/, PACKAGES/ to data/ (and remove RUNS/ as it should be the same as data/)

### Phase 2 — Engine and packaging

- [ ] Implement CLI parsing, mode selection, execution scheduling, worker limits,
      performance reporting, and post-run actions.
- [ ] Implement ZIP creation, SHA-256 hashing, package naming, metadata, and output
      relocation.
- [ ] Implement the MODS discovery and packaging contract.

### Phase 3 — Core collectors

- [ ] Implement system/hardware inventory and command-output collectors.
- [ ] Implement process, memory, filesystem, network, and adapter collectors.
- [ ] Implement registry, event-log, storage, encryption, Bluetooth, USB, Wi-Fi,
      SSH, browser, and media collectors.

### Phase 4 — Deep collection and visualization

- [ ] Implement sensitive-file mining, recursive directory/tree collection, and
      Sysinternals integration.
- [ ] Implement packet capture, CSV export, and network graph generation.
- [ ] Implement flag-history statistics and usage graph generation.

### Phase 5 — Maintenance and release readiness

- [ ] Implement debugger, update, developer checks, manifest comparison, and
      version handling.
- [ ] Add per-collector tests using mocked Windows command/registry/WMI responses.
- [ ] Add Windows integration tests for each output contract.
- [ ] Test default, threaded, minimal, deep, non-Python, performance, modded,
      debug, update, usage, shutdown, reboot, cancellation, and permission-denied
      flows.
- [ ] Verify that every collector is represented in the CLI mode matrix and the
      v4.0 documentation.
- [ ] Verify that packages contain generated evidence and metadata only.
- [ ] Verify that hashes reproduce from the final package bytes.
- [ ] Update README, contribution guidance, configuration documentation, and the
      repository wiki for v4.0.

## 9. v4.0 completion checklist

- [ ] Every feature in this file has an implementation owner/status.
- [ ] Every collector can be run directly and through the orchestrator.
- [ ] Every mode has an explicit collector inclusion matrix.
- [ ] Every output has a documented path, format, and retention rule.
- [x] A failed collector cannot silently make the run appear successful.
- [x] A cancelled run leaves a recoverable partial-run report and cleans only its
      own temporary data.
- [x] A normal run creates a package, a matching SHA-256 hash, and a run manifest.
- [ ] Debug/update/developer tools are separated from normal collection behavior.
- [ ] The final v4.0 release is documented as a complete recreation of the feature
      surface listed here.

Once all is complete, ask the user to git clone the wiki of logicytics so that we may update the DOCS completly
