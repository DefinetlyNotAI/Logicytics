# Core collector catalog

Core collectors are shipped, discovered automatically, and selected by a profile or exact `--include` ID. The table below is generated from the live `metadata()` declarations. `standard` means routine local inventory; `deep` means broader or more sensitive collection; `minimal`, `offline`, and `manual` identify narrower or explicitly selected entries. Capabilities are the access categories to review before authorization.

## System and hardware

| ID | Purpose | Profiles | Access | Output |
| --- | --- | --- | --- | --- |
| `core.system.bios_info` | BIOS manufacturer, name, version, release date | standard, deep | subprocess | HTML |
| `core.system.computer_system` | Computer model, manufacturer, processor count | standard, deep | subprocess | JSON |
| `core.system.defender_status` | Defender engine, signatures, protection state | standard, deep | subprocess | JSON |
| `core.system.environment_posture` | Administrator state, UAC, PowerShell policies | standard, deep | subprocess | JSON |
| `core.system.group_policy` | Group Policy Result summary | standard, deep | subprocess | text |
| `core.system.installed_drivers` | Windows driver inventory | standard, deep | subprocess | CSV |
| `core.system.installed_updates` | Installed hotfix IDs, descriptions, dates | standard, deep | subprocess | JSON |
| `core.system.local_accounts` | Local account names, SIDs, state, descriptions, login times | deep | subprocess | JSON |
| `core.system.operating_system` | Detailed Windows operating-system data | standard, deep | subprocess | JSON |
| `core.system.scheduled_tasks` | Bounded scheduled-task names, paths, authors, states | deep | subprocess | JSON |
| `core.system.session_snapshot` | User/SID, OS build, memory, language, host, time, drive | deep | subprocess | JSON |
| `core.system.system_details` | Windows `systeminfo` report | standard, deep | subprocess | text |
| `core.system.system_diagnostics` | Architecture, CPU, page size, boot time | standard, deep | subprocess | JSON |
| `core.system.system_info` | Bounded OS and hardware inventory | minimal, standard, deep, offline | none | JSON |
| `core.system.windows_services` | Service names, state, start mode, account, executable | standard, deep | subprocess | JSON |
| `core.system.windows_system_data_backup` | Bounded policy, event-log, and security-support evidence | deep | filesystem_read, sensitive_files | binary files |
| `core.system.wmic_inventory` | Optional legacy WMIC computer-system inventory | manual | subprocess | text |
| `core.hardware.battery_status` | Battery status, charge, capacity, runtime | standard, deep | subprocess | JSON |
| `core.hardware.display_adapters` | Display adapter, driver, resolution, memory | standard, deep | subprocess | JSON |
| `core.hardware.windows_features` | Optional-feature names and enabled states | standard, deep | subprocess | JSON |

## Storage, filesystem, and memory

| ID | Purpose | Profiles | Access | Output |
| --- | --- | --- | --- | --- |
| `core.storage.logical_drives` | Logical-drive types and aggregate capacity | minimal, standard, deep, offline | none | JSON |
| `core.storage.mounted_volumes` | Volume GUID and mount-point mappings | standard, deep | subprocess | text |
| `core.storage.physical_disks` | Disk model, media, interface, size | standard, deep | subprocess | JSON |
| `core.storage.volume_details` | Drive type, filesystem, label, capacity | standard, deep | none | JSON |
| `core.filesystem.startup_folder_entries` | Startup-folder names, locations, size, timestamps | deep | filesystem_read | JSON |
| `core.filesystem.system_drive_listing` | Bounded recursive system-drive listing | deep | filesystem_read | text |
| `core.filesystem.system_drive_tree` | Bounded recursive directory tree | deep | filesystem_read | text |
| `core.filesystem.sensitive_file_inventory` | Bounded matching sensitive filenames and copies | deep | filesystem_read, sensitive_files | binary files |
| `core.memory.memory_snapshot` | Physical, virtual, and page-file totals | minimal, standard, deep, offline | none | JSON |
| `core.process.memory_map` | Readable virtual-memory regions and RSS | deep | none | JSON |

## Processes and network

| ID | Purpose | Profiles | Access | Output |
| --- | --- | --- | --- | --- |
| `core.process.running_processes` | Non-verbose task list | standard, deep | subprocess | CSV |
| `core.process.detailed_processes` | Verbose task list | deep | subprocess | CSV |
| `core.process.process_memory` | Working-set, private, virtual counters | deep | subprocess | JSON |
| `core.network.active_connections` | TCP/UDP endpoints, state, owning PID | deep | subprocess | text |
| `core.network.adapter_statistics` | Interface bytes, packets, errors, discards | deep | subprocess | JSON |
| `core.network.arp_cache` | Local ARP cache | deep | subprocess | text |
| `core.network.bandwidth_sample` | Average and peak interface bandwidth | deep | subprocess | JSON |
| `core.network.connection_processes` | Netstat endpoints correlated with processes | deep | subprocess | CSV |
| `core.network.dns_cache` | Local DNS resolver cache | deep | subprocess | text |
| `core.network.firewall_profiles` | Domain, Private, Public firewall settings | standard, deep | subprocess | JSON |
| `core.network.network_adapters` | IP configuration and adapter details | standard, deep | subprocess | text |
| `core.network.network_identity` | Hostname and resolver-provided addresses | standard, deep | network | JSON |
| `core.network.network_interfaces` | IPv4, masks, link, speed, duplex | deep | subprocess | JSON |
| `core.network.routing_table` | IPv4 and IPv6 routes | deep | subprocess | text |
| `core.packet.connection_graph` | DOT source/destination graph with protocol labels | deep | subprocess | Graphviz DOT |
| `core.packet.packet_capture` | Bounded IPv4 packet metadata, no payloads | deep | network, packet_capture, elevated_privileges | CSV |

## Wireless, Bluetooth, USB, and registry

| ID | Purpose | Profiles | Access | Output |
| --- | --- | --- | --- | --- |
| `core.wireless.wifi_interfaces` | Wi-Fi state and normalized names | deep | subprocess | JSON |
| `core.wireless.wifi_profiles` | Saved Wi-Fi profile names | deep | subprocess | text |
| `core.wireless.wifi_profile_keys` | Saved profile XML including key material | deep | subprocess, sensitive_files | XML |
| `core.bluetooth.paired_devices` | Bluetooth PnP device metadata | deep | subprocess | JSON |
| `core.bluetooth.bluetooth_addresses` | Bluetooth names and address-like identifiers | deep | subprocess | JSON |
| `core.bluetooth.bluetooth_history` | Timestamped Bluetooth PnP snapshot | deep | subprocess | JSON |
| `core.usb.usb_storage_inventory` | USB storage class, ID, name, last-write time | deep | registry_read | JSON |
| `core.registry.installed_applications` | Names, versions, publishers, install metadata | standard, deep | registry_read | JSON |
| `core.registry.startup_applications` | User and machine Run/RunOnce entries | standard, deep | registry_read | JSON |
| `core.registry.hklm_backup` | HKLM hive backup | deep | registry_read, subprocess, sensitive_files | `.reg` text |

## Logs, encryption, browser, and integrations

| ID | Purpose | Profiles | Access | Output |
| --- | --- | --- | --- | --- |
| `core.event_log.application_events` | Up to 1,000 Application events | deep | subprocess | CSV |
| `core.event_log.security_events` | Up to 1,000 Security events | deep | subprocess | CSV |
| `core.event_log.system_events` | Up to 1,000 System events | deep | subprocess | CSV |
| `core.encryption.bitlocker_status` | Read-only drive encryption status | standard, deep | subprocess | text |
| `core.encryption.bitlocker_volumes` | Read-only BitLocker volume metadata | standard, deep | subprocess | JSON |
| `core.browser.browser_data_backup` | Bounded local Edge, Chrome, Firefox, Opera evidence | deep | filesystem_read, browser_data, sensitive_files | binary files |
| `core.media.media_backup` | Bounded Pictures/Videos JPG, PNG, MP4 copies | deep | filesystem_read, sensitive_files | binary files |
| `core.ssh.ssh_backup` | `.ssh` keys and configuration archive | deep | filesystem_read, sensitive_files, private_keys | ZIP |
| `core.diagnostics.sysinternals_report` | Sysinternals state and available tool output | deep | subprocess | text |
| `core.integration.legacy_code_outputs` | Bounded generated evidence from `CODE` | manual | filesystem_read | binary files |

All entries default to standard privilege and no remote network access except the packet and network-identity declarations shown above. A missing Windows facility is normally recorded as skipped with an actionable reason.
