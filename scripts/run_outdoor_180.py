#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive dual-view 'outdoor_shade' restyling over all 180 tictactoe episodes via the API.

Uses the API's default generation parameters (balanced controls, 15 steps, 720, sigma 110,
guidance 5) — only the style (outdoor_shade) and per-episode cell-specific prompt are set.
Resumable: skips episodes already succeeded (recorded in progress.json). One job at a time
(single GPU); ~45-50 min/episode at 720 -> ~6 days for 180.

Run (server must be up with COSMOS_API_OUTPUT_DIR=outputs/outdoor_180):
  uv run --no-sync python scripts/run_outdoor_180.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "http://localhost:8011"
TD = "/home/demouser/robotics/DellRobot/tictactoe_data/videos/chunk-000/observation.images"
OUT = Path("outputs/outdoor_180")
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"

# Episode -> destination cell (matches scripts/run_tictactoe_unified.py EPISODE_CELL_MAP).
CELL_MAP = [
    (range(0, 20), "bottom-left"), (range(20, 40), "middle-left"), (range(40, 60), "top-left"),
    (range(60, 80), "bottom-center"), (range(80, 100), "center"), (range(100, 120), "top-center"),
    (range(120, 140), "bottom-right"), (range(140, 160), "middle-right"), (range(160, 180), "top-right"),
]
PROMPT_FMT = (
    "A robot arm picks up a game piece and places it in the {cell} cell of a tic-tac-toe board on "
    "a table. The board rests on a plain matte surface with a simple, uniform dark background "
    "surrounding it. The game pieces are solid, non-glowing, matte markers with consistent colour "
    "and material throughout."
)


def cell_of(ep: int) -> str:
    for rng, cell in CELL_MAP:
        if ep in rng:
            return cell
    raise ValueError(ep)


def load() -> dict:
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def save(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=2))


def submit_with_retry(payload: dict, retries: int = 3) -> str | None:
    for i in range(retries):
        try:
            r = requests.post(f"{API}/generate/dual_view", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["job_id"]
        except Exception as e:
            print(f"  submit attempt {i + 1} failed: {e}", flush=True)
            time.sleep(30)
    return None


def main() -> None:
    progress = load()
    done = sum(1 for v in progress.values() if v.get("state") == "succeeded")
    print(f"Starting outdoor_shade run; {done}/180 already done.", flush=True)
    for ep in range(180):
        key = f"{ep:06d}"
        if progress.get(key, {}).get("state") == "succeeded":
            continue
        cell = cell_of(ep)
        payload = {
            "prompt": PROMPT_FMT.format(cell=cell),
            "top_path": f"{TD}.top/episode_{key}.mp4",
            "wrist_path": f"{TD}.wrist/episode_{key}.mp4",
            "styles": ["outdoor_shade"],
        }
        t0 = time.time()
        jid = submit_with_retry(payload)
        if jid is None:
            progress[key] = {"state": "submit_failed", "cell": cell}
            save(progress)
            print(f"[{ep + 1}/180] ep{key} {cell} -> SUBMIT FAILED", flush=True)
            continue

        state, st = "queued", {}
        deadline = time.time() + 7200
        while time.time() < deadline:
            try:
                st = requests.get(f"{API}/jobs/{jid}", timeout=20).json()
                state = st["state"]
            except Exception:
                time.sleep(20)
                continue
            if state in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(30)

        rec = {"job_id": jid, "state": state, "cell": cell, "wall_min": round((time.time() - t0) / 60, 1)}
        if state == "succeeded" and st.get("samples"):
            s = st["samples"][0]
            rec["views"] = s.get("view_outputs")
            rec["gen_s"] = s.get("generation_time_s")
        else:
            rec["error"] = st.get("error")
        progress[key] = rec
        save(progress)
        ndone = sum(1 for v in progress.values() if v.get("state") == "succeeded")
        print(f"[{ep + 1}/180] ep{key} {cell} -> {state}  ({rec['wall_min']}min)  total done={ndone}", flush=True)

    print("ALL EPISODES PROCESSED.", flush=True)


if __name__ == "__main__":
    main()
