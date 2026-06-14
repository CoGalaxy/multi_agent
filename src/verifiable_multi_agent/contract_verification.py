"""Research topic 1: structured contract verification.

This module is the readable entry point for the first thesis thread:
multi-agent communication is recorded as ContractMessage objects, then checked
by structural and optional semantic verification.
"""

from __future__ import annotations

from verifiable_multi_agent.contracts import ContractMessage, VerificationResult
from verifiable_multi_agent.verifier import SemanticVerifier, verify_contracts

__all__ = [
    "ContractMessage",
    "SemanticVerifier",
    "VerificationResult",
    "verify_contracts",
]
