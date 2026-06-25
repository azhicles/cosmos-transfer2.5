#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge an exp_run_batch results.json with a Reason2 critic scores.json into a table.

Usage:
  uv run --no-sync python scripts/exp_report.py --results outputs/experiments/batchN/results.json \
      --scores outputs/experiments/batchN/scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--scores", required=True, help="Reason2 critic --out JSON.")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    scores = json.loads(Path(args.scores).read_text())
    # critic output is a list of {video, scores:{task_score, realism_score}, combined}
    by_video = {Path(s["video"]).resolve().as_posix(): s for s in scores}

    rows = []
    for r in results:
        cfg = r.get("config", {})
        vp = r.get("video_path")
        sc = by_video.get(Path(vp).resolve().as_posix()) if vp else None
        ts = sc["scores"].get("task_score") if sc else None
        rs = sc["scores"].get("realism_score") if sc else None
        rows.append({
            "label": r["label"],
            "controls": ",".join(cfg.get("controls", ["depth", "seg", "edge"])),
            "steps": cfg.get("num_steps", 35),
            "res": cfg.get("resolution", "720"),
            "sigma": cfg.get("sigma_max", 110),
            "time_s": r.get("gen_time_s"),
            "vram_gb": r.get("vram_alloc_gb"),
            "task": ts,
            "realism": rs,
            "state": r.get("state"),
        })

    hdr = ["label", "controls", "steps", "res", "sigma", "time_s", "vram_gb", "task", "realism", "state"]
    widths = {h: max(len(h), *(len(str(row[h])) for row in rows)) for h in hdr} if rows else {h: len(h) for h in hdr}
    line = "  ".join(h.ljust(widths[h]) for h in hdr)
    print(line)
    print("  ".join("-" * widths[h] for h in hdr))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in hdr))


if __name__ == "__main__":
    main()
