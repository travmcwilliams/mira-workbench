# agents package - the 10 cooperating agents of the DSI platform
# All agents follow the conventions in spec §4:
#   - print column names + first 3 rows of input before parsing
#   - print final record count after writing
#   - request DB via utils.db.get_db(), never open own connections
# Explicit imports commented to avoid relative import issues when running
# scripts directly (common on Windows dev). Do direct submodule imports instead:
#   from agents.source_ingestor import ingest_by_id
#   from agents.schema_mapper import map_to_canonical
# (The individual modules themselves insert the project root into sys.path for robustness.)

