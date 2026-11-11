# SESSION_START.md
Operational protocol for the DSI platform (see spec §12).

## Session Start (VM assumed deallocated)
1. Start the VM:
   ```bash
   az vm start --name app-vm --resource-group <rg>   # or equivalent for your cloud
   ```
2. Connect your AI-assisted IDE (Cursor etc.) via **Remote SSH** to the VM.
   (Edits must land on the VM, not your laptop.)
3. On the VM:
   ```bash
   # Recommended aliases (put in ~/.bashrc):
   # alias app='cd /home/app/project && source .venv/bin/activate'
   # alias app-verify='python scripts/registry_repair.py --verify-only'
   # alias app-push='python -c "from utils.object_storage import push_db; push_db()"'
   # alias app-pull='python -c "from utils.object_storage import pull_db; pull_db()"'

   app
   git pull origin main
   app-verify
   ```

## Session End (always do this)
```bash
app-push          # push DB + raw to object storage (source of truth)
git add -A && git commit -m "describe the work" && git push origin main
az vm deallocate --name app-vm --resource-group <rg>   # stop paying for idle
```

## Local Windows Dev (no VM)
- Set env vars or use .env (see ENVIRONMENT.md)
- Run the same python commands.
- Use `python scripts/rebuild_db.py` to reset.
- For demo ingest: see scripts/run_demo_ingest.py (created by scaffold)

## Aliases you should configure
On VM and in docs:
- `app`
- `app-verify`
- `app-push`
- `app-pull`

## After code changes that affect DB
Always:
```bash
python scripts/rebuild_db.py
python scripts/registry_repair.py --verify-only
```

## Common commands
- Ingest one source (demo local file): `python -m agents.source_ingestor --source-id 00001 --local-file data/raw/00001/grants_alpha_2025.csv`
- (or the demo wrapper)
- Full registry health: `python scripts/registry_repair.py --verify-only`
