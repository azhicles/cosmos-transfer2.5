#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rerank experiment configs by style-aware critic scores and find the resource/quality Pareto frontier.

Groups all scored clips by recipe signature (controls, steps, res, sigma, vis weight), averages the
style-aware quality across seeds, joins per-config time + VRAM, ranks, and marks the Pareto-optimal
set (minimise time, maximise quality). Quality = style_score + realism_score (0-20).

Usage:
  uv run --no-sync python scripts/exp_pareto.py --scores outputs/experiments/rescore_styleaware.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def sig(cfg: dict) -> tuple:
    controls = tuple(sorted(cfg.get("controls", ["depth", "seg", "edge"])))
    vis_w = (cfg.get("control_weights") or {}).get("vis")
    if vis_w is None and "vis" in controls:
        vis_w = 0.3
    return (controls, cfg.get("num_steps"), str(cfg.get("resolution", "720")),
            str(cfg.get("sigma_max", 110)), vis_w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Style-aware critic output JSON.")
    args = ap.parse_args()

    scores = {}
    for s in json.load(open(args.scores)):
        scores[os.path.realpath(s["video"])] = s.get("scores", {})

    # video -> config (+ time/vram) from all batch results
    groups = defaultdict(lambda: {"style": [], "realism": [], "combined": [], "cfg": None,
                                   "time_min": None, "vram": None, "n": 0})
    for rj in sorted(glob.glob("outputs/experiments/*/results.json")):
        if "batch1_steps" in rj:
            continue
        for r in json.load(open(rj)):
            vp = r.get("video_path")
            if not vp:
                continue
            sc = scores.get(os.path.realpath(vp))
            if not sc:
                continue
            cfg = r.get("config", {})
            k = sig(cfg)
            g = groups[k]
            g["cfg"] = cfg
            g["time_min"] = round(r["gen_time_s"] / 60, 1) if r.get("gen_time_s") else None
            g["vram"] = r.get("vram_alloc_gb")
            st, rs = sc.get("style_score"), sc.get("realism_score")
            if st is not None:
                g["style"].append(st)
            if rs is not None:
                g["realism"].append(rs)
            if st is not None and rs is not None:
                g["combined"].append(st + rs)
            g["n"] += 1

    rows = []
    for k, g in groups.items():
        controls, steps, res, sigma, vis_w = k
        rows.append({
            "controls": "+".join(c for c in ["edge", "depth", "seg", "vis"] if c in controls),
            "steps": steps, "res": res, "sigma": sigma, "vis": vis_w, "n": g["n"],
            "time_min": g["time_min"], "vram": g["vram"],
            "style": round(statistics.mean(g["style"]), 1) if g["style"] else None,
            "realism": round(statistics.mean(g["realism"]), 1) if g["realism"] else None,
            "quality": round(statistics.mean(g["combined"]), 1) if g["combined"] else None,
        })

    # Pareto frontier: minimise time_min, maximise quality.
    valid = [r for r in rows if r["quality"] is not None and r["time_min"] is not None]
    for r in valid:
        r["pareto"] = not any(
            o is not r and o["time_min"] <= r["time_min"] and o["quality"] >= r["quality"]
            and (o["time_min"] < r["time_min"] or o["quality"] > r["quality"])
            for o in valid
        )

    rows.sort(key=lambda r: (-(r["quality"] or -1), r["time_min"] or 1e9))
    hdr = ["pareto", "controls", "steps", "res", "sigma", "vis", "n", "time_min", "vram", "style", "realism", "quality"]
    w = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    print("  ".join("-" * w[h] for h in hdr))
    for r in rows:
        r["pareto"] = "★" if r.get("pareto") else ""
        print("  ".join(str(r.get(h, "")).ljust(w[h]) for h in hdr))

    print("\n★ = Pareto-optimal (no other config is both faster AND higher quality).")
    front = sorted([r for r in valid if r["pareto"]], key=lambda r: r["time_min"])
    print("\nFrontier (fastest → best):")
    for r in front:
        print(f"  {r['time_min']:>5} min  {r['vram']:>5} GB  quality={r['quality']:<4} "
              f"(style {r['style']} / realism {r['realism']})  {r['controls']} · {r['steps']}st · {r['res']}p · vis{r['vis']}")


if __name__ == "__main__":
    main()
