# ENVIRONMENT.md
Standardized environment + path conventions for the DSI platform (per spec §2).

## Path Rules (never violate)
- All code reads paths via:
  ```python
  PROJECT_ROOT = Path(os.environ.get("APP_PROJECT_ROOT", "/home/app/project"))
  DB_PATH      = Path(os.environ.get("APP_DB_PATH", str(PROJECT_ROOT / "db/app.duckdb")))
  ```
- No hardcoded OS-specific paths anywhere.
- On Windows dev machine: set the two APP_* env vars (or rely on .env + code defaults for this checkout).
- On Linux VM: the /home/app/project defaults are authoritative.

## Python
- Target: Python 3.10+ (3.11/3.12 preferred for wheels).
- On VM: install the exact minor version explicitly (`sudo apt install python3.11 ...` or pyenv).
- Always use venv on VM.

## Requirements
- `requirements.txt` : local dev (your workstation OS)
- `requirements_vm.txt` : Linux production image / VM (pinned, no dev-only packages)
- After `pip install -r requirements_vm.txt` on VM: `playwright install chromium`

## Secrets
1. Create managed vault.
2. Assign secrets-officer role on that vault.
3. Wait 2 min for propagation.
4. Verify with `secret list` (or equivalent) BEFORE writing any code that calls get_secret.
5. Write secrets.
6. ALSO `export STORAGE-CONNECTION-STRING=...` (and other critical) in `~/.bashrc` on the VM.
   Shell profile wins over .env for connection strings.

## Object Storage
- Container naming convention: `dsi-db` (for app.duckdb), `dsi-raw` (for archived downloads).
- Use the pinned azure-storage-blob version.

## VM Lifecycle
See SESSION_START.md and spec §12.

## Git
- Clone with PAT (never password).
- After any registry or data change: `python scripts/registry_repair.py --verify-only && git status`

## First time on a new machine / VM
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on win
pip install -r requirements_vm.txt   # or _vm on Linux
playwright install chromium
python scripts/rebuild_db.py
python scripts/registry_repair.py --verify-only
```
