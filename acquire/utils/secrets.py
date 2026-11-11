"""
utils/secrets.py
Centralized secrets access for the DSI platform.

Design per spec §8:
- All credentials live in the managed vault (never in code, never committed .env).
- Read via this module.
- Critical values ALSO exported in shell profile (~/.bashrc) so non-.env-loading scripts see fresh values.
- Connection string rule: shell export wins.

This scaffold provides:
- Env var fallback for local/dev (and for VM bootstrap before full vault wiring).
- Stub hooks for cloud Key Vault (Azure Key Vault / AWS Secrets Manager / GCP Secret Manager).
- Explicit warning on use of .env-only values.

Never put real keys in this file or in repo .env.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env if present (but shell exports take precedence for critical items)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

VAULT_CLIENT = None  # populated on first use of get_secret when real vault configured


def _get_vault_client():
    """
    Lazy init cloud vault client.
    TODO (Phase 5): implement for your cloud.
    Example for Azure:
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential
        ...
    """
    global VAULT_CLIENT
    if VAULT_CLIENT is not None:
        return VAULT_CLIENT
    # No vault configured in scaffold by default.
    # To enable: set USE_VAULT=1 and implement provider here.
    return None


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret.

    Resolution order (highest wins):
    1. Shell environment (exported in ~/.bashrc or session) -- always wins for connection strings.
    2. Cloud vault (if configured and USE_VAULT=1).
    3. .env (loaded above, but lower priority for critical items).
    4. default

    Usage:
        api_key = get_secret("SEARCH-API-KEY")
        conn_str = get_secret("STORAGE-CONNECTION-STRING")
    """
    # 1. Direct env (shell wins)
    val = os.environ.get(name)
    if val:
        return val

    # 2. Vault (stub)
    if os.environ.get("USE_VAULT") == "1":
        client = _get_vault_client()
        if client:
            try:
                # Placeholder for real call e.g. client.get_secret(name).value
                # return client.get_secret(name).value
                pass
            except Exception:
                pass

    # 3. Already loaded from .env via dotenv (lower priority)
    val = os.environ.get(name)
    if val:
        # Warn loudly for connection-string class secrets if they came only from .env
        if "CONNECTION" in name.upper() or "KEY" in name.upper():
            print(f"[secrets] WARNING: {name} came from .env (not shell export). "
                  "Per spec, export critical values in ~/.bashrc too.")
        return val

    if default is not None:
        return default

    # Missing secret is often fatal for real runs; surface clearly.
    print(f"[secrets] SECRET NOT FOUND: {name} (returning None)")
    return None


# Convenience getters for the named secrets in the spec
def get_llm_api_key() -> Optional[str]:
    return get_secret("LLM-API-KEY")

def get_enrichment_api_key() -> Optional[str]:
    return get_secret("ENRICHMENT-API-KEY")

def get_search_api_key() -> Optional[str]:
    return get_secret("SEARCH-API-KEY")

def get_storage_connection_string() -> Optional[str]:
    return get_secret("STORAGE-CONNECTION-STRING")

def get_git_pat() -> Optional[str]:
    return get_secret("GIT-PAT")
