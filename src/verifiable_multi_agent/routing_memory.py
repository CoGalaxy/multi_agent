"""Research topic 3: route reuse and failure-driven topology upgrade.

ProtocolMemory stores historical routing outcomes in the two-dimensional
TaskProfile space. The quant router uses nearby failures to upgrade topology
complexity when a simple route has been unreliable before.
"""

from __future__ import annotations

from verifiable_multi_agent.memory import MemoryRecord, ProtocolMemory

__all__ = [
    "MemoryRecord",
    "ProtocolMemory",
]
