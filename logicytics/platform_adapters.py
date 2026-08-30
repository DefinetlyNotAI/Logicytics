"""Injectable, mockable boundaries for host-specific operations used by collectors."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any


class ProcessAdapter:
    """Run one explicit, shell-free host command through a central policy seam."""

    def run(self, command: Sequence[str], **options: Any) -> subprocess.CompletedProcess[str]:
        """Delegate to the guarded stdlib runner while retaining its familiar result contract."""
        normalized = tuple(str(argument) for argument in command)
        if not normalized:
            raise ValueError("command must contain at least one argument")
        if options.get("shell") is True:
            raise ValueError("collector process adapters never permit shell execution")
        timeout = options.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise ValueError("command timeout must be positive")
        return subprocess.run(normalized, **options)


process_adapter = ProcessAdapter()
