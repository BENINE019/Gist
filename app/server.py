#!/usr/bin/env python3
"""Mimic demo server.

Serves the single-page UI, exposes the results already in out/<clip>/, runs a
fresh shuffled scene on demand, and runs the whole pipeline on an uploaded
phone clip as a background job.

Run from the repo root:
  MUJOCO_GL=egl uvicorn app.server:app --host 0.0.0.0 --port 8000
then open http://localhost:8000

Requires: pip install fastapi uvicorn python-multipart
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
CLIPS = ROOT / "clips"
STATIC = Path(__file__).resolve().parent / "static"
MARKER_CM = float(os.environ.get("MARKER_CM", "9.5"))

sys.path.insert(0, str(ROOT))
from mimic import planner  # noqa: E402
from mimic.generalize import shuffle  # noqa: E402
from mimic.intent import edit_goal  # noqa: E402
from mimic.sim import Scene, fit_to_reach, load_clip, plan_from_events, scene_objects  # noqa: E402

app = FastAPI(title="Mimic")
app.mount("/out", StaticFiles(directory=str(OUT)), name="out")
app.mount("/clips", StaticFiles(directory=str(CLIPS)), name="clips")

JOBS: dict[str, dict] = {}
SIM_LOCK = threading.Lock()


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def clip_summary(name: str) -> dict | None:
    d = OUT / name
    intent = read(d / "intent.json")
    if not intent:
        return None
    gen = read(d / "generalize" / "summary.json") or {}
    return {"name": name, "task": intent.get("task_sentence", ""),
            "goal": intent["candidates"][0].get("goal", "") if intent.get("candidates") else "",
            "goal_shuffled": gen.get("goal_success_shuffled"), "copy_shuffled": gen.get("copy_success_shuffled"),
            "episodes": gen.get("episodes_exported"), "video": f"/clips/{name}.mp4" if (CLIPS / f"{name}.mp4").exists() else None}


def rendered(scene_dir: Path, which: str) -> str | None:
    p = scene_dir / which / "episode.mp4"
    return f"/out/{p.relative_to(OUT).as_posix()}" if p.exists() else None


@app.get("/api/clips")
def api_clips():
    names = sorted(p.name for p in OUT.glob("c[0-9][0-9]") if (p / "intent.json").exists())
    names += sorted(p.name for p in OUT.glob("u_*") if (p / "intent.json").exists())
    return [s for s in (clip_summary(n) for n in names) if s]


@app.get("/api/clip/{name}")
def api_clip(name: str):
    d = OUT / name
    if not (d / "events.json").exists():
        raise HTTPException(404, "no results for this clip")
    gen = d / "generalize"
    scenes = []
    if gen.exists():
        summ = read(gen / "summary.json") or {}
        for row in summ.get("scenes", []):
            sd = gen / f"scene_{row['scene']:02d}"
            scenes.append({**row, "goal_video": rendered(sd, "goal"), "copy_video": rendered(sd, "copy")})
    return {
        "name": name,
        "video": f"/clips/{name}.mp4" if (CLIPS / f"{name}.mp4").exists() else None,
        "names": read(d / "names.json"),
        "events": read(d / "events.json"),
        "narration": read(d / "narration.json"),
        "intent": read(d / "intent.json"),
        "intent_no_narration": read(d / "intent_no_narration.json"),
        "topdown": f"/out/{name}/topdown.png" if (d / "topdown.png").exists() else None,
        "timeline": f"/out/{name}/timeline.png" if (d / "timeline.png").exists() else None,
        "replay": rendered(d / "sim", "replay"),
        "scenes": scenes,
        "summary": read(gen / "summary.json"),
    }


@app.post("/api/shuffle/{name}")
def api_shuffle(name: str, extra: bool = False, distractor: bool = False, body: dict | None = Body(None)):
    """One fresh shuffled scene, both robots rendered. Takes 15 to 60 seconds.
    Optional JSON body {"goal": {...}} runs that goal instead of the inferred one."""
    d = OUT / name
    intent = read(d / "intent.json")
    if not intent or not intent.get("candidates"):
        raise HTTPException(404, "no goal for this clip")
    goal = (body or {}).get("goal") or intent["candidates"][0]
    with SIM_LOCK:
        tt, ev = load_clip(d)
        base = scene_objects(tt, ev)
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        for _ in range(30):
            objects = shuffle(base, rng, extra, distractor)
            already, _ = planner.check(goal, objects, {o["key"]: o["xy"] for o in objects})
            if not already:
                break
        steps_copy = plan_from_events(ev, objects)
        fit_to_reach(objects, [steps_copy])
        steps_goal, note = planner.plan(goal, objects)
        live = d / "generalize" / "live"
        shutil.rmtree(live, ignore_errors=True)
        (live / "goal").mkdir(parents=True)
        (live / "copy").mkdir(parents=True)
        rg = Scene(objects, render=True).execute(steps_goal, live / "goal")
        rc = Scene(objects, render=True).execute(steps_copy, live / "copy")
    ok_g, why_g = planner.check(goal, objects, rg["final_xy_cm"])
    ok_c, why_c = planner.check(goal, objects, rc["final_xy_cm"])
    stamp = int(time.time())
    return {"goal": goal, "plan_note": note, "objects": objects,
            "goal_ok": bool(ok_g), "goal_why": why_g, "goal_steps": rg["steps"],
            "copy_ok": bool(ok_c), "copy_why": why_c, "copy_steps": rc["steps"],
            "goal_video": f"/out/{name}/generalize/live/goal/episode.mp4?t={stamp}",
            "copy_video": f"/out/{name}/generalize/live/copy/episode.mp4?t={stamp}"}


@app.post("/api/edit/{name}")
def api_edit(name: str, body: dict = Body(...)):
    """Edit the goal with a plain-language instruction. Body: {"instruction": "...", "goal": {...} (optional)}."""
    d = OUT / name
    intent = read(d / "intent.json")
    events = read(d / "events.json")
    if not intent or not events:
        raise HTTPException(404, "no goal for this clip")
    instruction = (body.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(400, "empty instruction")
    base = body.get("goal") or intent["candidates"][0]
    try:
        res = edit_goal(base, instruction, events["objects"])
    except Exception as e:  # surface the model error to the page
        raise HTTPException(502, f"goal edit failed: {e}")
    scene = scene_objects(*load_clip(d))
    steps, note = planner.plan(res["goal"], scene)
    res["plan_note"] = note
    res["plan_steps"] = len(steps)
    return res


def run_pipeline(job: dict, src: Path, name: str) -> None:
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    mp4 = CLIPS / f"{name}.mp4"
    wav = CLIPS / "audio" / f"{name}.wav"
    wav.parent.mkdir(exist_ok=True)
    steps = [
        ("Converting video", ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", "scale=-2:720,fps=30",
                              "-c:v", "libx264", "-crf", "23", "-an", str(mp4)]),
        ("Extracting narration", ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)]),
        ("Naming objects (Muse Spark)", [sys.executable, "scripts/name_objects.py", str(mp4), "--output", f"out/{name}"]),
        ("Tracking (SAM 3.1)", [sys.executable, "-m", "mimic.tracking", str(mp4), "--names", f"out/{name}/names.json"]),
        ("Mapping to the table", [sys.executable, "-m", "mimic.table_map", str(mp4), "--marker-cm", str(MARKER_CM)]),
        ("Finding events", [sys.executable, "-m", "mimic.events", str(mp4)]),
        ("Transcribing narration (Muse Voice)", [sys.executable, "-m", "mimic.transcribe", str(mp4)]),
        ("Inferring the goal (Muse Spark)", [sys.executable, "-m", "mimic.intent", str(mp4)]),
        ("Replaying and shuffling (MuJoCo)", [sys.executable, "-m", "mimic.generalize", str(mp4), "--scenes", "3", "--render", "2"]),
    ]
    for i, (label, cmd) in enumerate(steps):
        job.update(stage=label, progress=i / len(steps))
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0 and "transcribe" not in " ".join(cmd):
            job.update(status="failed", stage=label, error=(r.stderr or r.stdout)[-1500:])
            return
    job.update(status="done", stage="Done", progress=1.0, clip=name)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    name = f"u_{job_id}"
    raw = CLIPS / "raw"
    raw.mkdir(exist_ok=True)
    src = raw / f"{name}{Path(file.filename or 'clip.mp4').suffix or '.mp4'}"
    src.write_bytes(await file.read())
    job = {"id": job_id, "status": "running", "stage": "Queued", "progress": 0.0, "clip": None}
    JOBS[job_id] = job
    threading.Thread(target=run_pipeline, args=(job, src, name), daemon=True).start()
    return job


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
