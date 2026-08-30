"""Archive the current user's SSH directory after explicit sensitive approval."""

from __future__ import annotations

import zipfile
from pathlib import Path

from logicytics import Capability, CollectorMetadata, CollectorResult, CoreCollector, EvidenceKind, Specialty, ValidationResult
from logicytics.contracts import CollectorContext, CollectorStatus
from logicytics.platform_adapters import filesystem_adapter

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_SOURCE_BYTES = 128 * 1024 * 1024


class SshBackupCollector(CoreCollector):
    """Back up current-user SSH keys and configuration into the private workspace."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the explicit-consent SSH archive artifact contract."""
        return CollectorMetadata(
            id="core.ssh.ssh_backup", name="SSH directory backup", version="4.0.0", specialty=Specialty.SSH,
            output_media_types=("application/zip",),
            description="Archives the current user's .ssh keys and configuration after explicit sensitive-data approval.",
            author="Logicytics", supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ, Capability.SENSITIVE_FILES, Capability.PRIVATE_KEYS),
            sensitive_data_categories=("private_keys", "ssh_configuration", "credentials"), default_profiles=("deep",),
            timeout_seconds=120, maximum_output_bytes=128 * 1024 * 1024,
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Check cancellation state before scanning the current user's SSH directory."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Archive bounded regular files from .ssh while preserving relative paths."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before SSH backup")
        ssh_directory = filesystem_adapter.home() / ".ssh"
        try:
            directory_exists = ssh_directory.is_dir()
        except OSError as error:
            return CollectorResult(CollectorStatus.SKIPPED, "the current user's .ssh directory is inaccessible",
                                   errors=(str(error),))
        if not directory_exists:
            return CollectorResult(CollectorStatus.SKIPPED, "the current user has no .ssh directory")
        context.report_progress("ssh_backup_started")
        archive = context.workspace / "ssh_backup.zip"
        source_bytes = 0
        archived_files = 0
        skipped_files = 0
        cancelled = False
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            try:
                candidates = sorted(filesystem_adapter.recursive(ssh_directory))
            except OSError as error:
                return CollectorResult(CollectorStatus.SKIPPED, "the current user's .ssh directory is inaccessible",
                                       errors=(str(error),))
            for candidate in candidates:
                if context.is_cancelled:
                    cancelled = True
                    break
                try:
                    is_file = candidate.is_file()
                    is_symlink = candidate.is_symlink()
                    size = candidate.stat().st_size if is_file else 0
                except OSError:
                    skipped_files += 1
                    continue
                if not is_file or is_symlink:
                    continue
                if size > MAX_FILE_BYTES or source_bytes + size > MAX_ARCHIVE_SOURCE_BYTES:
                    skipped_files += 1
                    continue
                output.write(candidate, arcname=candidate.relative_to(ssh_directory).as_posix())
                if context.is_cancelled:
                    cancelled = True
                    break
                source_bytes += size
                archived_files += 1
        if cancelled:
            archive.unlink(missing_ok=True)
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled during SSH backup")
        if archived_files == 0:
            return CollectorResult(CollectorStatus.SKIPPED, "no SSH files met the bounded backup policy")
        if context.is_cancelled:
            archive.unlink(missing_ok=True)
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before SSH-backup registration")
        artifact = context.artifacts.register_file(
            archive,
            media_type="application/zip",
            evidence_kind=EvidenceKind.RAW,
            transformations=("archived from the current user's SSH directory",),
        )
        context.report_progress(
            "ssh_backup_finished", archived_files=archived_files, skipped_files=skipped_files,
            source_bytes=source_bytes, bytes_written=artifact.size_bytes,
        )
        return CollectorResult.succeeded("SSH directory backup collected", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        """Leave archive removal to the isolated collector workspace lifecycle."""
