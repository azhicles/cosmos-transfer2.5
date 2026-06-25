#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal-requirements experiment harness for the Cosmos Transfer API.

Generates one short single-view clip per config (episode top view, reusing pre-extracted
control videos so there is NO control-generation cost), recording wall time and peak GPU VRAM
for each. Output videos + a results.json are written under --out. Score them afterwards with the
Reason2 critic (the GPU can't host both the diffusion server and the 32B critic at once).

Each config (JSON list via --configs) supports:
  label, num_steps, resolution, controls (subset of ["edge","depth","seg"]), sigma_max,
  guidance, num_video_frames_per_chunk, style (inline {name,suffix} or a bundled name).

Usage:
  uv run --no-sync python scripts/exp_run_batch.py \
      --api http://localhost:8011 --top-video /abs/ep47_top.mp4 \
      --controls-dir outputs/experiments/controls_ep47_top \
      --configs '[{"label":"steps4","num_steps":4}, ...]' \
      --base-prompt "..." --negative "..." --out outputs/experiments/batchN
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

DEFAULT_STYLE = {"name": "studio_warm",
                 "suffix": "Warm cinematic studio lighting with a soft key light and gentle rim "
                           "light, rich natural true-to-life colours. Photorealistic."}


def parse_vram(metrics_log: Path) -> dict:
    """Pull the last sample's peak VRAM (alloc/reserved GB) from a job metrics.log."""
    out = {"vram_alloc_gb": None, "vram_reserved_gb": None}
    if not metrics_log.is_file():
        return out
    for line in metrics_log.read_text().splitlines():
        m = re.search(r"VRAM peak alloc\s*:\s*([\d.]+)\s*GB", line)
        if m:
            out["vram_alloc_gb"] = float(m.group(1))
        m = re.search(r"VRAM peak reserved\s*:\s*([\d.]+)\s*GB", line)
        if m:
            out["vram_reserved_gb"] = float(m.group(1))
    return out


def build_controls(keys: list[str], controls_dir: Path, weight_override: dict | None = None) -> dict:
    weights = {"depth": 1.0, "seg": 1.0, "edge": 0.2, "vis": 0.3}
    if weight_override:
        weights = {**weights, **weight_override}
    controls = {}
    for k in keys:
        spec = {"control_weight": weights.get(k, 1.0)}
        cpath = controls_dir / f"{k}.mp4"
        if cpath.is_file():
            spec["control_path"] = str(cpath.resolve())
        controls[k] = spec
    return controls


def run_config(api: str, cfg: dict, top_video: str, controls_dir: Path, base_prompt: str,
               negative: str, out_dir: Path, poll_s: float = 20.0, timeout_s: float = 7200.0) -> dict:
    style = cfg.get("style", DEFAULT_STYLE)
    keys = cfg.get("controls", ["depth", "seg", "edge"])
    # Per-config overrides: a different input video, and/or its own prompt (for other episodes).
    video = cfg.get("top_video", top_video)
    # reuse=False -> point at an empty dir so controls regenerate on-the-fly for this input.
    cdir = controls_dir if cfg.get("reuse_controls", True) else Path("/nonexistent_controls")
    payload = {
        "prompt": cfg.get("prompt", base_prompt),
        "negative_prompt": negative,
        "video_path": video,
        "styles": [style],
        "num_steps": cfg.get("num_steps", 35),
        "guidance": cfg.get("guidance", 5),
        "sigma_max": str(cfg.get("sigma_max", 110)),
        "resolution": str(cfg.get("resolution", "720")),
        "num_video_frames_per_chunk": cfg.get("num_video_frames_per_chunk", 93),
        "controls": build_controls(keys, cdir, cfg.get("control_weights")),
    }
    if cfg.get("max_frames") is not None:
        payload["max_frames"] = cfg["max_frames"]
    if cfg.get("seed") is not None:
        payload["seed"] = cfg["seed"]
    if cfg.get("guided_generation"):
        payload["guided_generation"] = True
        if cfg.get("guided_foreground_prompt"):
            payload["guided_foreground_prompt"] = cfg["guided_foreground_prompt"]
        if cfg.get("guided_generation_step_threshold") is not None:
            payload["guided_generation_step_threshold"] = cfg["guided_generation_step_threshold"]
    label = cfg["label"]
    t0 = time.monotonic()
    r = requests.post(f"{api}/generate", json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + timeout_s
    state = "queued"
    while time.monotonic() < deadline:
        st = requests.get(f"{api}/jobs/{job_id}").json()
        state = st["state"]
        if state in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(poll_s)
    wall = time.monotonic() - t0

    result = {"label": label, "config": cfg, "job_id": job_id, "state": state,
              "wall_s": round(wall, 1), "video": None, "gen_time_s": None,
              "vram_alloc_gb": None, "vram_reserved_gb": None}
    if state == "succeeded":
        sample = st["samples"][0]
        result["gen_time_s"] = sample.get("generation_time_s")
        # Output lives at <server output root>/<job_id>/<output_file>; resolved by the caller.
        result["video"] = sample.get("output_file")
    else:
        result["error"] = st.get("error")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8011")
    ap.add_argument("--top-video", required=True)
    ap.add_argument("--controls-dir", required=True)
    ap.add_argument("--configs", required=True, help="JSON list of configs (or @path).")
    ap.add_argument("--base-prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--server-output-root", required=True, help="Server COSMOS_API_OUTPUT_DIR (to resolve job dirs).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    configs_raw = args.configs
    if configs_raw.startswith("@"):
        configs_raw = Path(configs_raw[1:]).read_text()
    configs = json.loads(configs_raw)
    controls_dir = Path(args.controls_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_root = Path(args.server_output_root)

    results = []
    for cfg in configs:
        print(f"[exp] running {cfg['label']} ...", flush=True)
        res = run_config(args.api, cfg, args.top_video, controls_dir, args.base_prompt,
                         args.negative, out_dir)
        # Resolve absolute video path from the server output root.
        if res.get("video"):
            res["video_path"] = str((server_root / res["job_id"] / res["video"]).resolve())
        # Pull VRAM from that job's metrics.log.
        res.update(parse_vram(server_root / res["job_id"] / "metrics.log"))
        print(f"[exp] {res['label']}: state={res['state']} gen_time={res.get('gen_time_s')}s "
              f"vram={res.get('vram_alloc_gb')}GB video={res.get('video_path')}", flush=True)
        results.append(res)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"[exp] wrote {out_dir/'results.json'} ({len(results)} configs)", flush=True)


if __name__ == "__main__":
    main()
