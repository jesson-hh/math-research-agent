"""Centralized configuration — reads from environment with sensible defaults."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
API_KEY = os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = (os.environ.get("BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL", "")).strip()
MODEL = os.environ.get("MODEL") or os.environ.get("ANTHROPIC_MODEL", "")
PROOF_MODEL = os.environ.get("PROOF_MODEL", "").strip() or None

# ── Timeouts (seconds) ──
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
SYMPY_TIMEOUT = int(os.environ.get("SYMPY_TIMEOUT", "30"))
CODE_TIMEOUT_MAX = int(os.environ.get("CODE_TIMEOUT_MAX", "30"))

# ── Agent ──
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
PROOF_MAX_TOKENS = int(os.environ.get("PROOF_MAX_TOKENS", "4096"))
CONTEXT_THRESHOLD = int(os.environ.get("CONTEXT_THRESHOLD", "80000"))

# ── ArXiv ──
ARXIV_DELAY = float(os.environ.get("ARXIV_DELAY", "3.0"))
ARXIV_RETRIES = int(os.environ.get("ARXIV_RETRIES", "3"))

# ── Autonomous research ──
AUTO_MAX_ITERATIONS = int(os.environ.get("AUTO_MAX_ITERATIONS", "20"))
AUTO_MAX_TIME = int(os.environ.get("AUTO_MAX_TIME", "600"))
AUTO_MAX_API_CALLS = int(os.environ.get("AUTO_MAX_API_CALLS", "50"))

# ── Server ──
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7861"))
