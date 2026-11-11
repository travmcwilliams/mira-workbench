"""
utils/object_storage.py
Push / pull DB file + raw archive to durable cloud object storage.

Per spec:
- Source of truth for the DB file + archive of raw downloads.
- Push/pull on session boundaries (see operational protocol §12).
- Pin the cloud SDK version in requirements_vm.txt.

This scaffold provides a minimal interface + Azure Blob stub.
Swap in AWS S3 / GCS by implementing the same functions.

Critical: the STORAGE-CONNECTION-STRING must come from shell export (see secrets.py).
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from .db import DB_PATH, PROJECT_ROOT
from .secrets import get_storage_connection_string

# Raw archive root
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

def _get_blob_service_client():
    """
    Return Azure BlobServiceClient or None.
    TODO: implement real auth (DefaultAzureCredential preferred over conn string for prod).
    Connection string is accepted for bootstrap.
    """
    conn_str = get_storage_connection_string()
    if not conn_str:
        return None
    try:
        # Pinned version required: azure-storage-blob==12.19.0 (example that works)
        from azure.storage.blob import BlobServiceClient
        return BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        print(f"[object_storage] Azure init failed: {e}")
        return None


def push_db(backup_name: Optional[str] = None) -> bool:
    """
    Upload the current app.duckdb to object storage container 'dsi-db'.
    Also upload a timestamped backup copy.
    """
    if not DB_PATH.exists():
        print("[object_storage] No local DB to push.")
        return False

    client = _get_blob_service_client()
    if client is None:
        # Dev fallback: copy to a local "remote" simulation under data/remote_backup
        backup_dir = PROJECT_ROOT / "data" / "remote_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / (backup_name or DB_PATH.name)
        shutil.copy2(DB_PATH, target)
        print(f"[object_storage] (DEV FALLBACK) DB copied to {target}")
        return True

    try:
        container = client.get_container_client("dsi-db")
        # Create container if needed (best effort)
        try:
            container.get_container_properties()
        except Exception:
            try:
                client.create_container("dsi-db")
            except Exception:
                pass
        container.upload_blob(name="app.duckdb", data=DB_PATH.read_bytes(), overwrite=True)
        print("[object_storage] DB pushed to cloud container dsi-db/app.duckdb")
        return True
    except Exception as e:
        print(f"[object_storage] Push DB failed: {e}")
        return False


def pull_db() -> bool:
    """Download the latest app.duckdb from storage into local DB_PATH (overwrites)."""
    client = _get_blob_service_client()
    if client is None:
        # Dev fallback
        backup = PROJECT_ROOT / "data" / "remote_backup" / "app.duckdb"
        if backup.exists():
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, DB_PATH)
            print(f"[object_storage] (DEV) DB pulled from fallback {backup}")
            return True
        print("[object_storage] No cloud DB and no dev fallback present.")
        return False

    try:
        container = client.get_container_client("dsi-db")
        blob = container.download_blob("app.duckdb")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DB_PATH, "wb") as f:
            f.write(blob.readall())
        print("[object_storage] DB pulled from cloud.")
        return True
    except Exception as e:
        print(f"[object_storage] Pull DB failed: {e}")
        return False


def archive_raw(source_id: str, local_path: Path) -> bool:
    """
    Archive a downloaded raw file under raw/{source_id}/... in cloud (and local).
    If the file is already at the target archive location (common when using fetcher),
    we skip the redundant copy to avoid Windows file-locking issues.
    """
    if not local_path.exists():
        return False
    dest = RAW_ROOT / source_id / local_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Avoid copying a file to itself (causes PermissionError on Windows when file is "in use")
    if local_path.resolve() == dest.resolve():
        print(f"[object_storage] Raw already archived at {local_path}")
    else:
        try:
            shutil.copy2(local_path, dest)
            print(f"[object_storage] Raw archived locally to {dest}")
        except Exception as e:
            print(f"[object_storage] Failed to copy raw file: {e}")
            return False

    # TODO: also upload to cloud blob prefix f"data/raw/{source_id}/{local_path.name}"
    client = _get_blob_service_client()
    if client:
        try:
            container = client.get_container_client("dsi-raw")
            blob_name = f"raw/{source_id}/{local_path.name}"
            container.upload_blob(name=blob_name, data=local_path.read_bytes(), overwrite=True)
            print(f"[object_storage] Raw uploaded to dsi-raw/{blob_name}")
        except Exception as e:
            print(f"[object_storage] Cloud raw upload skipped/failed: {e}")
    return True


def ensure_raw_dir(source_id: str) -> Path:
    p = RAW_ROOT / source_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_raw_tree(source_id: str, local_folder: Optional[Path] = None) -> list[str]:
    """
    Upload everything under data/raw/{source_id} (or provided folder) to cloud.
    Returns list of uploaded blob names.
    """
    folder = local_folder or (RAW_ROOT / source_id)
    if not folder.exists():
        return []
    uploaded = []
    client = _get_blob_service_client()
    if client is None:
        print("[object_storage] (DEV) raw tree sync skipped (no cloud client)")
        return uploaded
    try:
        container = client.get_container_client("dsi-raw")
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(folder)
                blob_name = f"raw/{source_id}/{rel.as_posix()}"
                container.upload_blob(name=blob_name, data=full.read_bytes(), overwrite=True)
                uploaded.append(blob_name)
        print(f"[object_storage] Uploaded {len(uploaded)} raw files for {source_id}")
    except Exception as e:
        print(f"[object_storage] raw tree upload failed: {e}")
    return uploaded


def download_raw_tree(source_id: str, dest_folder: Optional[Path] = None) -> bool:
    """Download raw files for a source from cloud into local raw dir."""
    dest = dest_folder or (RAW_ROOT / source_id)
    dest.mkdir(parents=True, exist_ok=True)
    client = _get_blob_service_client()
    if client is None:
        print("[object_storage] (DEV) raw tree download skipped")
        return False
    try:
        container = client.get_container_client("dsi-raw")
        prefix = f"raw/{source_id}/"
        for blob in container.list_blobs(name_starts_with=prefix):
            rel = blob.name[len(prefix):]
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = container.get_blob_client(blob.name).download_blob().readall()
            target.write_bytes(data)
        print(f"[object_storage] Downloaded raw tree for {source_id}")
        return True
    except Exception as e:
        print(f"[object_storage] raw tree download failed: {e}")
        return False
