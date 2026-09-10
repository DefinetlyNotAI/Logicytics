"""Integration-style checks for the v4 core without shipped collectors."""

from __future__ import annotations

import sys
from typing import Any

from logicytics.contracts import ResourceClass

COLLECTOR = """\
\"\"\"Create a harmless test artifact.\"\"\"

from pathlib import Path

from logicytics import CollectorMetadata, CollectorResult, CoreCollector, Specialty, ValidationResult
from logicytics import EstimatedCost, NetworkAccess, PrivilegeLevel
from logicytics.contracts import CollectorContext


class SystemInfoCollector(CoreCollector):
    \"\"\"A valid test core collector.\"\"\"

    @classmethod
    def metadata(cls) -> CollectorMetadata:
        \"\"\"Return test metadata.\"\"\"
        return CollectorMetadata(
            id="core.system.system_info",
            name="System info",
            version="1.0.0",
            specialty=Specialty.SYSTEM,
            output_media_types=("text/plain",),
            description="Creates a harmless text artifact for core tests.",
            author="tests",
            supported_platforms=("win32",),
            capabilities=(),
        )

    def validate(self, context: CollectorContext) -> ValidationResult:
        \"\"\"Validate the test collector.\"\"\"
        return ValidationResult(True)

    def collect(self, context: CollectorContext) -> CollectorResult:
        \"\"\"Write and register a harmless artifact.\"\"\"
        output = context.workspace / "system.txt"
        output.write_text("ok\\n", encoding="utf-8")
        artifact = context.artifacts.register_file(output, media_type="text/plain")
        return CollectorResult.succeeded("test artifact created", (artifact,))

    def cleanup(self, context: CollectorContext) -> None:
        \"\"\"Release test resources.\"\"\"
"""


def plugin_collector_source() -> str:
    """Convert the core fixture into a plugin with every security field explicit."""
    return COLLECTOR.replace("CoreCollector", "PluginCollector").replace(
        '            author="tests",',
        '            author="tests",\n'
        "            privilege_level=PrivilegeLevel.STANDARD,\n"
        "            sensitive_data_categories=(),\n"
        "            network_access=NetworkAccess.NONE,\n"
        "            estimated_cost=EstimatedCost.LOW,\n"
        "            timeout_seconds=60,\n"
        "            maximum_output_bytes=100 * 1024 * 1024,\n"
        '            minimum_contract_version="4.0",',
    )


def mod_metadata(name: str, *, filesystem_write: bool = False) -> dict[str, Any]:
    """Return a complete sidecar declaration for a harmless legacy script fixture."""
    return {
        "id": f"mod.{name}",
        "name": f"{name} mod",
        "version": "1.0.0",
        "specialty": "integration",
        "description": "Harmless isolated legacy script fixture.",
        "author": "tests",
        "supported_platforms": [sys.platform],
        "capabilities": [
            "subprocess",
            *(["filesystem_write"] if filesystem_write else []),
        ],
        "privilege_level": "standard",
        "sensitive_data_categories": [],
        "network_access": "none",
        "estimated_cost": "low",
        "timeout_seconds": 15,
        "maximum_output_bytes": 1024 * 1024,
        "maximum_artifact_files": 10,
        "output_media_types": ["text/plain"],
        "minimum_contract_version": "4.0",
        "default_profiles": ["standard"],
    }


def delayed_collector_source(
        filename: str,
        delay: float,
        *,
        parallel_safe: bool = True,
        dependencies: tuple[str, ...] = (),
        resource_class: ResourceClass = ResourceClass.GENERAL,
        fail: bool = False,
) -> str:
    """Create a valid fixture collector with observable scheduling duration."""
    class_name = "".join(part.title() for part in filename.split("_"))
    source = COLLECTOR.replace("SystemInfoCollector", f"{class_name}Collector")
    source = source.replace("core.system.system_info", f"core.system.{filename}")
    source = source.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom time import sleep\n",
    )
    source = source.replace(
        "from logicytics import CollectorMetadata,",
        "from logicytics import CollectorMetadata, ResourceClass,",
    )
    source = source.replace(
        '            supported_platforms=("win32",),',
        '            supported_platforms=("win32",),\n'
        f"            dependencies={dependencies!r},\n"
        f"            resource_class=ResourceClass.{resource_class.name},\n"
        f"            parallel_safe={parallel_safe!r},",
    )
    source = source.replace(
        "    def validate(self, context: CollectorContext) -> ValidationResult:\n",
        "    @classmethod\n"
        "    def dependencies(cls) -> tuple[str, ...]:\n"
        '        """Return fixture dependencies for scheduler tests."""\n'
        f"        return {dependencies!r}\n\n"
        "    def validate(self, context: CollectorContext) -> ValidationResult:\n",
    )
    source = source.replace(
        '        output = context.workspace / "system.txt"',
        f'        sleep({delay})\n        output = context.workspace / "system.txt"',
    )
    if fail:
        source = source.replace(
            f'        sleep({delay})\n        output = context.workspace / "system.txt"',
            f'        sleep({delay})\n        raise RuntimeError("dependency fixture failed")',
        )
    return source
