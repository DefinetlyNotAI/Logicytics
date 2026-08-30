"""Import generated evidence from historical CODE checkouts into the v4 artifact catalog."""

from __future__ import annotations

import shutil
from pathlib import Path

from logicytics import (
    Capability,
    CollectorMetadata,
    CollectorResult,
    CoreCollector,
    EvidenceKind,
    Specialty,
    ValidationResult,
)
from logicytics.contracts import CollectorContext, CollectorStatus

_EVIDENCE_EXTENSIONS = {
    ".csv", ".dot", ".evtx", ".html", ".htm", ".json", ".log", ".reg",
    ".svg", ".txt", ".xml", ".zip",
}
_EXCLUDED_DIRECTORIES = {
    ".git", ".idea", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__",
    "lib", "libs", "site-packages", "venv",
}
_MAXIMUM_FILES = 500
_MAXIMUM_FILE_BYTES = 64 * 1024 * 1024


class LegacyCodeOutputsCollector(CoreCollector):
    """Import bounded generated files while excluding executable project material."""

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        """Declare the compatibility evidence import and its filesystem-read boundary."""
        return CollectorMetadata(
            id="core.integration.legacy_code_outputs",
            name="Legacy CODE generated outputs",
            version="4.0.0",
            specialty=Specialty.INTEGRATION,
            output_media_types=("application/octet-stream",),
            description="Imports bounded generated evidence from CODE without packaging source or configuration.",
            author="Logicytics",
            supported_platforms=("win32",),
            capabilities=(Capability.FILESYSTEM_READ,),
            sensitive_data_categories=("legacy_generated_evidence",),
            default_profiles=("deep",),
            timeout_seconds=120,
            maximum_output_bytes=512 * 1024 * 1024,
            maximum_artifact_bytes=_MAXIMUM_FILE_BYTES,
            maximum_artifact_files=_MAXIMUM_FILES,
        )

    @staticmethod
    def _code_root() -> Path:
        """Resolve the historical directory relative to this checked-in collector source."""
        return Path(__file__).resolve().parents[2] / "CODE"

    def validate(self, context: CollectorContext) -> ValidationResult:
        """Skip cleanly when the historical CODE directory is not present."""
        if context.is_cancelled:
            return ValidationResult(False, reasons=("run cancellation was requested",))
        code_root = self._code_root()
        if not code_root.is_dir() or code_root.is_symlink():
            return ValidationResult(False, reasons=("historical CODE directory is unavailable",))
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        """Stream allowlisted generated files into the private workspace and artifact store."""
        if context.is_cancelled:
            return CollectorResult(CollectorStatus.CANCELLED, "cancelled before legacy CODE import")
        code_root = self._code_root().resolve()
        candidates: list[Path] = []
        for path in sorted(code_root.rglob("*")):
            if context.is_cancelled:
                return CollectorResult.cancelled("cancelled during legacy CODE discovery")
            relative = path.relative_to(code_root)
            if any(part.casefold() in _EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            try:
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.resolve().parent != (code_root / relative.parent).resolve()
                    or path.suffix.casefold() not in _EVIDENCE_EXTENSIONS
                    or path.stat().st_size > _MAXIMUM_FILE_BYTES
                ):
                    continue
            except OSError:
                continue
            candidates.append(path)
            if len(candidates) >= _MAXIMUM_FILES:
                break

        artifacts = []
        errors: list[str] = []
        for source in candidates:
            if context.is_cancelled:
                return CollectorResult.cancelled("cancelled during legacy CODE import", tuple(artifacts))
            relative = source.relative_to(code_root)
            destination = context.workspace / "legacy_code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                with source.open("rb") as source_stream, temporary.open("xb") as destination_stream:
                    while block := source_stream.read(1024 * 1024):
                        if context.is_cancelled:
                            raise InterruptedError("run cancellation was requested")
                        destination_stream.write(block)
                temporary.replace(destination)
                artifacts.append(
                    context.artifacts.register_file(
                        destination,
                        media_type="application/octet-stream",
                        evidence_kind=EvidenceKind.RAW,
                        transformations=("imported from historical CODE output",),
                    )
                )
            except InterruptedError:
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                return CollectorResult.cancelled("cancelled during legacy CODE import", tuple(artifacts))
            except OSError as error:
                temporary.unlink(missing_ok=True)
                errors.append(f"{relative.as_posix()}: {error}")
        if not artifacts and not errors:
            return CollectorResult.skipped("no generated legacy CODE evidence matched the import policy")
        if errors:
            return CollectorResult.partial(
                "legacy CODE evidence imported with file-level failures",
                tuple(artifacts),
                errors=tuple(errors),
            )
        context.report_progress(
            "legacy_code_outputs_finished",
            imported_files=len(artifacts),
            bytes_written=sum(artifact.size_bytes for artifact in artifacts),
        )
        return CollectorResult.succeeded("legacy CODE evidence imported", tuple(artifacts))

    def cleanup(self, context: CollectorContext) -> None:
        """Remove only an unpublished temporary copy left by an interrupted import."""
        for temporary in context.workspace.rglob("*.tmp"):
            temporary.unlink(missing_ok=True)
