from __future__ import annotations

"""FastAPI entry‑point exposing:

POST /run          – kick off optimisation in a background task
GET  /job/{id}      – poll job status / retrieve results metadata
Static /results/*   – serve PNGs (map + graphs) generated per job

A janitor coroutine deletes job folders and metadata older than
48 hours so the VPS cannot fill up.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core import run_job
from app import models as m

# ----------------------------------------------------------------------------
# Configuration constants – tweak as needed
# ----------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent / "app" / "data"      # ↳ read‑only datasets
RESULTS_DIR = Path(__file__).parent / "results"  # ↳ runtime artefacts
RESULTS_DIR.mkdir(exist_ok=True)

RETENTION_SEC = 48 * 3600        # delete jobs after 2 days
JANITOR_INTERVAL = 3600          # run cleanup once an hour

# ----------------------------------------------------------------------------
# Globals (okay for <100 jobs/day). Swap for Redis when you scale.
# ----------------------------------------------------------------------------

JOBS: Dict[str, m.JobStatus] = {}

# ----------------------------------------------------------------------------
# FastAPI setup
# ----------------------------------------------------------------------------

app = FastAPI(title="Sensor-Optimisation API", version="1.0.0")

# Allow dev server + prod domain. Adjust as required.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],  # open by default
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve /results/<job_id>/<png>
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# ----------------------------------------------------------------------------
# Background worker
# ----------------------------------------------------------------------------
def _report(job_id: str, *, progress: int | None = None, step: str | None = None, detail: str | None = None, append_log: str | None = None):
    job = JOBS.get(job_id)
    if not job:
        return
    if progress is not None:
        job.progress = max(0, min(100, int(progress)))
    if step is not None:
        job.step = step
    if detail is not None:
        job.detail = detail
    if append_log:
        if not getattr(job, "logs", None):
            job.logs = []
        job.logs.append(append_log)


def _run_and_store(job_id: str, req: m.JobRequest):
    """Wrapper executed in a BackgroundTasks thread."""

    try:
        output_dir = RESULTS_DIR / job_id
        _report(job_id, progress=1, step="starting", detail="Starting job…")

        result = run_job(
            city=req.city,
            # opti_method=req.opti_method,
            timeframe=req.timeframe,
            n_sensors=req.n_sensors,
            data_root=DATA_ROOT,
            output_dir=output_dir,
            report=lambda pct, msg=None, step=None: _report(
                job_id, progress=pct, detail=msg, step=step
            ),
        )

        JOBS[job_id].status = "finished"  # type: ignore[assignment]
        JOBS[job_id].result = m.JobResult(**result)
        _report(job_id, progress=100, step="done", detail="Finished.")
    except Exception as exc:  # pylint: disable=broad-except
        JOBS[job_id].status = "error"      # type: ignore[assignment]
        JOBS[job_id].detail = str(exc)
    finally:
        # Mark expiry so janitor eventually deletes.
        JOBS[job_id].expires_at = time.time() + RETENTION_SEC  # type: ignore[assignment]

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.post("/run", response_model=m.JobStatus)
def run_endpoint(req: m.JobRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    job_status = m.JobStatus(job_id=job_id, status="running", started=time.time())
    JOBS[job_id] = job_status

    bg.add_task(_run_and_store, job_id, req)
    return job_status


@app.get("/job/{job_id}", response_model=m.JobStatus)
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# Convenience: download a single artefact directly
@app.get("/job/{job_id}/{filename}")
def download(job_id: str, filename: str):
    job = JOBS.get(job_id)
    if not job or job.status != "finished":
        raise HTTPException(status_code=404)
    file_path = RESULTS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(file_path)

# ----------------------------------------------------------------------------
# Janitor – kills old jobs + artefacts
# ----------------------------------------------------------------------------

def _cleanup():
    now = time.time()
    for job_id, meta in list(JOBS.items()):
        expiry = meta.expires_at or (meta.started + RETENTION_SEC)
        if now > expiry:
            # Delete disk folder (ignore if already gone)
            folder = RESULTS_DIR / job_id
            try:
                for p in folder.glob("**/*"):
                    p.unlink(missing_ok=True)
                folder.rmdir()
            except FileNotFoundError:
                pass
            # Drop from memory
            JOBS.pop(job_id, None)


async def _janitor_loop():
    while True:
        _cleanup()
        await asyncio.sleep(JANITOR_INTERVAL)


@app.on_event("startup")
async def _start_janitor() -> None:
    asyncio.create_task(_janitor_loop())

# ----------------------------------------------------------------------------
# CLI entry so `python main.py` just works in dev.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
