"""Compatibility entry point for contract verification."""

from __future__ import annotations

from verifiable_multi_agent.contracts import ContractMessage, VerificationResult
from verifiable_multi_agent.verifier import ContractVerifier, SemanticVerifier, verify_contracts

__all__ = [
    "ContractMessage",
    "ContractVerifier",
    "SemanticVerifier",
    "VerificationResult",
    "verify_contracts",
]
