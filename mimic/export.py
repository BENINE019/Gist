#!/usr/bin/env python3
"""Export verified episodes as a LeRobot v2.1 dataset.

Collects every successful goal run under out/<clip>/generalize/scene_*/goal/
(for one clip or all clips), re-renders the ones that were run without video,
and writes:

  dataset/<name>/
    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet ...
    videos/chunk-000/observation.images.phone/episode_000000.mp4 ...

Each frame: observation.state = 7 arm joint angles + gripper opening,
action = the next frame's state (kinematic replay), plus timestamp and indices.
The task string is the sentence Muse Spark wrote for the demonstration.

Usage (from the repo root):
  python -m mimic.export --name mimic_demo                  # all clips
  python -m mimic.export --name c01_only --clips c01
  python -m mimic.export --name quick --clips c01 --max-per-clip 3

Requires: pip install pandas pyarrow ; ffmpeg on PATH for video re-encoding.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

STATE_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
VIDEO_KEY = "observation.images.phone"


def episodes_for_clip(out_dir: Path, max_per_clip: int | None) -> list[dict]:
    gen = out_dir / "generalize"
    if not gen.exists():
        return []
    eps = []
    for scene_dir in sorted(gen.glob("scene_*")):
        res = scene_dir / "goal" / "result.json"
        if not res.exists():
            continue
        r = json.loads(res.read_text())
        if not r.get("goal_satisfied") or not r.get("trajectory"):
            continue
        eps.append({"clip": out_dir.name, "scene_dir": scene_dir, "result": r})
        if max_per_clip and len(eps) >= max_per_clip:
            break
    return eps


def ensure_video(ep: dict, video: Path) -> Path | None:
    """Return a rendered episode.mp4 for this episode, rendering it if needed."""
    goal_dir = ep["scene_dir"] / "goal"
    mp4 = goal_dir / "episode.mp4"
    if mp4.exists():
        return mp4
    # re-run with rendering, same scene and plan
    from . import planner
    from .sim import Scene
    scene = json.loads((ep["scene_dir"] / "scene.json").read_text())
    steps, _ = planner.plan(scene["goal"], scene["objects"])
    sc = Scene(scene["objects"], render=True)
    res = sc.execute(steps, goal_dir)
    ok, why = planner.check(scene["goal"], scene["objects"], res["final_xy_cm"])
    res["goal_satisfied"], res["why"] = bool(ok), why
    (goal_dir / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    ep["result"] = res
    return mp4 if mp4.exists() else None


def reencode(src: Path, dst: Path, fps: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        shutil.copy(src, dst)
        return False
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-r", str(fps),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-an", str(dst)]
    subprocess.run(cmd, check=True)
    return True


def stats_for(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    return {"min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(),
            "mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(), "count": [int(a.shape[0])]}


def main() -> None:
    import pandas as pd

    p = argparse.ArgumentParser(description="Export verified episodes as a LeRobot v2.1 dataset")
    p.add_argument("--name", required=True, help="dataset folder name under dataset/")
    p.add_argument("--clips", nargs="*", help="clip names (default: every out/c*)")
    p.add_argument("--max-per-clip", type=int)
    p.add_argument("--out-root", type=Path, default=Path("out"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true", help="skip videos (parquet and meta only)")
    args = p.parse_args()

    clip_dirs = [args.out_root / c for c in args.clips] if args.clips else sorted(args.out_root.glob("c[0-9][0-9]"))
    episodes = []
    for d in clip_dirs:
        episodes += episodes_for_clip(d, args.max_per_clip)
    if not episodes:
        p.error("no successful goal episodes found; run mimic.generalize first")

    root = Path("dataset") / args.name
    for sub in ("meta", "data/chunk-000", f"videos/chunk-000/{VIDEO_KEY}"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    ep_rows, stat_rows = [], []
    total_frames = 0
    global_index = 0
    video_info = None

    for ei, ep in enumerate(episodes):
        intent = json.loads((args.out_root / ep["clip"] / "intent.json").read_text())
        task = intent.get("task_sentence") or ep["result"].get("why", "demonstrated task")
        task_index = tasks.setdefault(task, len(tasks))

        if not args.no_video:
            mp4 = ensure_video(ep, Path())
            if mp4 is not None:
                dst = root / "videos/chunk-000" / VIDEO_KEY / f"episode_{ei:06d}.mp4"
                reencode(mp4, dst, args.fps)
                if video_info is None:
                    import cv2
                    cap = cv2.VideoCapture(str(dst))
                    video_info = {"video.fps": args.fps, "video.height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  "video.width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "video.channels": 3,
                                  "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                  "video.is_depth_map": False, "has_audio": False}
                    cap.release()

        traj = ep["result"]["trajectory"]
        state = np.array([t["q"] + [t["gripper"]] for t in traj], dtype=np.float32)
        action = np.vstack([state[1:], state[-1:]])
        n = len(state)
        df = pd.DataFrame({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(n) / args.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        })
        df.to_parquet(root / "data/chunk-000" / f"episode_{ei:06d}.parquet", index=False)

        scene = json.loads((ep["scene_dir"] / "scene.json").read_text())
        ep_rows.append({"episode_index": ei, "tasks": [task], "length": n,
                        "source_clip": ep["clip"], "scene_variant": scene.get("variant", ""),
                        "goal": intent["candidates"][0].get("goal", "") if intent.get("candidates") else ""})
        stat_rows.append({"episode_index": ei, "stats": {
            "observation.state": stats_for(state), "action": stats_for(action),
            "timestamp": stats_for(df["timestamp"].to_numpy()),
            "frame_index": stats_for(df["frame_index"].to_numpy()),
            "episode_index": stats_for(df["episode_index"].to_numpy()),
            "index": stats_for(df["index"].to_numpy()),
            "task_index": stats_for(df["task_index"].to_numpy()),
        }})
        total_frames += n
        global_index += n
        print(f"episode {ei:03d}  {ep['clip']}  {ep['scene_dir'].name:<9} {n:>4} frames  {task}")

    features = {
        "observation.state": {"dtype": "float32", "shape": [8], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [8], "names": STATE_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if video_info is not None:
        features[VIDEO_KEY] = {"dtype": "video", "shape": [video_info["video.height"], video_info["video.width"], 3],
                               "names": ["height", "width", "channels"], "info": video_info}
    info = {
        "codebase_version": "v2.1", "robot_type": "panda_sim_kinematic",
        "total_episodes": len(episodes), "total_frames": total_frames, "total_tasks": len(tasks),
        "total_videos": len(episodes) if video_info is not None else 0, "total_chunks": 1, "chunks_size": 1000,
        "fps": args.fps, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "mimic": {"note": "Kinematic replay of goal plans inferred from single phone demonstrations; "
                          "scripted grasp; success verified against simulator state."},
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    with (root / "meta/episodes.jsonl").open("w") as f:
        for r in ep_rows:
            f.write(json.dumps(r) + "\n")
    with (root / "meta/tasks.jsonl").open("w") as f:
        for task, ti in tasks.items():
            f.write(json.dumps({"task_index": ti, "task": task}) + "\n")
    with (root / "meta/episodes_stats.jsonl").open("w") as f:
        for r in stat_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(episodes)} episodes, {total_frames} frames, {len(tasks)} task(s) -> {root}")
    print("Check: python -c \"import pandas as pd; print(pd.read_parquet('%s').head())\"" % (root / "data/chunk-000/episode_000000.parquet"))


if __name__ == "__main__":
    main()
