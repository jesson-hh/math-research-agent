"""Proof / technique knowledge base for cross-paper retrieval.

v1.8: every deeply distilled paper produces a `proof_sidecar` JSON blob with
extracted theorems / definitions / techniques. We store these in a SQLite +
FTS5 database alongside the vault so that subsequent distillations can
retrieve relevant prior theorems and inject them into the LLM context.

Storage layout (per-vault):
    <vault_path>/.proof_store/proofs.db        SQLite + FTS5
    <vault_path>/.proof_store/<arxiv_id>.json  Per-paper sidecar archive
"""

from .store import ProofStore, Theorem, Technique, ProofSidecar

__all__ = ["ProofStore", "Theorem", "Technique", "ProofSidecar"]
