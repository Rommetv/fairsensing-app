# new models
from __future__ import annotations

"""Pydantic schemas shared by *main.py* (FastAPI) and *core.py*.

Keeping every request/response shape here avoids circular imports and makes
pytest-based validation easy.
"""

from typing import Literal, Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Type aliases / literals that map exactly to the UI dropdowns or slider
# ----------------------------------------------------------------------------

City = Literal["Amsterdam", "Rotterdam", "The Hague", "Utrecht"]  # Extend once other cities are supported.
Timeframe = Literal["1day", "3day", "7day"]
StatusStr = Literal["running", "finished", "error"]

# ----------------------------------------------------------------------------
# Request coming from React ➜ /run (POST)
# ----------------------------------------------------------------------------

class JobRequest(BaseModel):
    """Shape of the JSON payload posted by the React form."""

    city: City = Field(
        ..., description="Target city for the optimisation run", example="Amsterdam"
    )
    timeframe: Timeframe = Field(
        ..., description="Historical window the grouped GPKG is built on",
        example="1day",
    )
    n_sensors: int = Field(
        ..., ge=1, le=500, description="Number of sensors to select", example=10
    )

# ----------------------------------------------------------------------------
# Internal representation + response for /job/{id}
# ----------------------------------------------------------------------------

class ArtefactSet(BaseModel):
    map_html: str

class TopVehicles(BaseModel):
    spatial: List[str]
    temporal: List[str]
    fair: List[str]
    population: List[str]

class JobResult(BaseModel):
    job_id: str
    top_vehicles: TopVehicles
    artefacts: Dict[str, ArtefactSet]
    stats_table: Dict[str, Any]
    geojson: Optional[Dict[str, Any]] = None
    graph_data: Optional[Dict[str, Dict[str, Any]]] = None

class JobStatus(BaseModel):
    """What the client polls for progress."""

    job_id: str
    status: StatusStr
    started: float
    result: Optional[JobResult] = None
    detail: Optional[str] = None
    expires_at: Optional[float] = None  # filled by janitor logic
    progress: Optional[int] = 0           # 0..100
    step: Optional[str] = None            # short step id/name
    logs: Optional[List[str]] = None
