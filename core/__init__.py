"""Core package for DS Workbench (project management, data, pipeline orchestration)."""

from .data_loader import detect_schema, load_file, load_from_upload  # noqa: F401
from .library_catalog import (  # noqa: F401
    LIBRARIES,
    for_stage,
    get_enabled_libraries,
    get_task_config,
    pycaret_models_for,
    set_enabled_libraries,
    set_task_config,
)
from .pipeline import Pipeline, StageResult  # noqa: F401
from .project import Project  # noqa: F401
