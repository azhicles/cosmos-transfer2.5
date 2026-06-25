#!/usr/bin/env python3
"""
Unified dual-view (top + wrist) Cosmos Transfer inference for the tictactoe dataset.

Both camera views are combined into a single video and processed in one inference pass,
guaranteeing frame-level synchronisation in style, timing, and action.

STACKING MODES
  Default (safe): vstack — top view above wrist view (640×960 combined).
    Detected as 3:4 aspect ratio → processed at 704×960 internally.
    Same latent token count as single-view inference. VRAM: ~73 GB (fits RTX 6000 96 GB).

  --full-res:     hstack — top view left of wrist view (1280×480 combined).
    Detected as 16:9 → processed at 1280×704. ~33% more latent tokens.
    Estimated VRAM: ~86–97 GB. May OOM. If it does, omit --full-res or set
    "resolution": "480" in inference_config.json to reduce internal resolution.

CONTROL VIDEO CACHING
  On-the-fly control videos (edge, depth, seg, vis) are cached per episode under
  ../tictactoe_data/control_cache/<mode>/episode_NNNNNN/ and reused on subsequent
  runs of the same episode in the same mode.

PROMPT CUSTOMISATION
  Edit assets/tictactoe/unified/style_prompt.txt to change the visual style.
  The {cell_position} placeholder is substituted per episode — do not remove it.

USAGE (run from the cosmos-transfer2.5 directory):
    python scripts/run_tictactoe_unified.py -o outputs/unified
    python scripts/run_tictactoe_unified.py -o outputs/unified --episodes 0-9
    python scripts/run_tictactoe_unified.py -o outputs/unified --num-episodes 5
    python scripts/run_tictactoe_unified.py -o outputs/unified --full-res
    python scripts/run_tictactoe_unified.py -o outputs/unified --num-steps 20 --seed 42
    python scripts/run_tictactoe_unified.py -o outputs/unified --image-context-path assets/tictactoe/image_style/reference.jpg
    python scripts/run_tictactoe_unified.py -o outputs/unified --force
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Prevent CUDA allocator from hoarding fragmented blocks across episodes.
# expandable_segments lets PyTorch grow/shrink VMM segments on demand;
# max_split_size_mb caps the largest block the caching allocator will split.
# Must be set before any CUDA context is created.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_EPISODES = 180
TOP_DIR   = Path("../tictactoe_data/videos/chunk-000/observation.images.top")
WRIST_DIR = Path("../tictactoe_data/videos/chunk-000/observation.images.wrist")

CONFIG_DIR          = Path("assets/tictactoe/unified")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "inference_config.json"
DEFAULT_PROMPT_PATH = CONFIG_DIR / "style_prompt.txt"
DEFAULT_CACHE_ROOT  = Path("../tictactoe_data/control_cache")

# Must match cosmos_transfer2.config.CONTROL_KEYS order for batch_hint_keys sorting.
CONTROL_KEYS = ["edge", "vis", "depth", "seg"]

# Episode index → destination cell description (substituted into {cell_position}).
EPISODE_CELL_MAP = [
    (range(  0,  20), "bottom-left cell"),
    (range( 20,  40), "middle-left cell"),
    (range( 40,  60), "top-left cell"),
    (range( 60,  80), "bottom-center cell"),
    (range( 80, 100), "center cell"),
    (range(100, 120), "top-center cell"),
    (range(120, 140), "bottom-right cell"),
    (range(140, 160), "middle-right cell"),
    (range(160, 180), "top-right cell"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_cell_position(episode_idx: int) -> str:
    for ep_range, cell in EPISODE_CELL_MAP:
        if episode_idx in ep_range:
            return cell
    raise ValueError(f"Episode index {episode_idx} must be in [0, {NUM_EPISODES - 1}]")


def parse_episodes(episodes_str: str | None, num_episodes: int | None) -> list[int]:
    """Resolve --episodes / --num-episodes into a sorted list of episode indices."""
    if episodes_str is not None:
        if episodes_str.lower() == "all":
            return list(range(NUM_EPISODES))
        m = re.fullmatch(r"(\d+)-(\d+)", episodes_str.strip())
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start > end:
                raise ValueError(f"Invalid range: {start} > {end}")
            return list(range(start, end + 1))
        try:
            return sorted(set(int(x.strip()) for x in episodes_str.split(",")))
        except ValueError:
            raise ValueError(
                f"Cannot parse --episodes '{episodes_str}'. "
                "Use: all | 0-9 | 0,5,10"
            )
    if num_episodes is not None:
        return list(range(min(num_episodes, NUM_EPISODES)))
    return list(range(NUM_EPISODES))


def run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def create_combined_video(
    top_path: Path, wrist_path: Path, out_path: Path, full_res: bool
) -> None:
    """Stack top and wrist into one combined video.

    vstack (default): 640×960. Detected as 3:4, processed at 704×960 — identical
    latent count to single-view 4:3 processing. VRAM-safe on 96 GB RTX 6000.

    hstack (--full-res): 1280×480. Detected as 16:9, processed at 1280×704 —
    ~33% more latent tokens. Estimated peak VRAM ~86–97 GB. May OOM.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_str = "hstack=inputs=2" if full_res else "vstack=inputs=2"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(top_path),
        "-i", str(wrist_path),
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(out_path),
    ])


def split_combined_video(
    combined_path: Path, top_out: Path, wrist_out: Path, full_res: bool
) -> None:
    """Split the combined output back into individual top and wrist view files."""
    if full_res:
        # hstack: left half = top, right half = wrist
        top_crop   = "crop=iw/2:ih:0:0"
        wrist_crop = "crop=iw/2:ih:iw/2:0"
    else:
        # vstack: top half = top, bottom half = wrist
        top_crop   = "crop=iw:ih/2:0:0"
        wrist_crop = "crop=iw:ih/2:0:ih/2"

    run_ffmpeg(["ffmpeg", "-y", "-i", str(combined_path),
                "-filter:v", top_crop,   "-loglevel", "error", str(top_out)])
    run_ffmpeg(["ffmpeg", "-y", "-i", str(combined_path),
                "-filter:v", wrist_crop, "-loglevel", "error", str(wrist_out)])


def get_cached_controls(
    ep_idx: int, cache_root: Path, mode_key: str, active_controls: list[str]
) -> dict[str, Path]:
    """Return {ctrl: Path} for every control video found in the cache."""
    ep_cache = cache_root / mode_key / f"episode_{ep_idx:06d}"
    return {
        ctrl: ep_cache / f"{ctrl}.mp4"
        for ctrl in active_controls
        if (ep_cache / f"{ctrl}.mp4").exists()
    }


def save_controls_to_cache(
    ep_dir: Path,
    ep_idx: int,
    cache_root: Path,
    mode_key: str,
    active_controls: list[str],
    already_cached: dict[str, Path],
) -> None:
    """Copy freshly generated control videos to the cache (skip already-cached ones)."""
    ep_cache = cache_root / mode_key / f"episode_{ep_idx:06d}"
    ep_cache.mkdir(parents=True, exist_ok=True)
    for ctrl in active_controls:
        if ctrl in already_cached:
            continue
        src = ep_dir / f"control_{ctrl}.mp4"
        if src.exists():
            dst = ep_cache / f"{ctrl}.mp4"
            shutil.copy2(src, dst)
            print(f"         [cache] {ctrl} → {dst.relative_to(Path.cwd())}")


def build_sample_dict(
    ep_idx: int,
    combined_path: Path,
    prompt: str,
    config: dict,
    cached_controls: dict[str, Path],
    active_controls: list[str],
) -> dict:
    """Build the dict that InferenceArguments.model_validate() consumes."""
    sample: dict = {
        "name":                   f"episode_{ep_idx:06d}_combined",
        "video_path":             str(combined_path.resolve()),
        "prompt":                 prompt,
        "guidance":               config.get("guidance", 3),
        "num_steps":              config.get("num_steps", 35),
        "seed":                   config.get("seed", 2025),
        "num_video_frames_per_chunk": config.get("num_video_frames_per_chunk", 93),
        "keep_input_resolution":  config.get("keep_input_resolution", True),
        "resolution":             config.get("resolution", "720"),
    }

    for ctrl in active_controls:
        ctrl_cfg = dict(config[ctrl])
        if ctrl in cached_controls:
            ctrl_cfg["control_path"] = str(cached_controls[ctrl].resolve())
        sample[ctrl] = ctrl_cfg

    image_ctx = config.get("image_context_path")
    if image_ctx:
        sample["image_context_path"] = str(Path(image_ctx).resolve())

    return sample


def write_run_summary(run_dir: Path, processed: list[int], total_elapsed: float) -> None:
    n = len(processed)
    lines = [
        "",
        "=" * 50,
        f" RUN SUMMARY",
        f"   Episodes completed   : {n}",
    ]
    if n:
        lines.append(
            f"   Range               : {processed[0]}–{processed[-1]}"
            if n > 1
            else f"   Episode             : {processed[0]}"
        )
        lines += [
            f"   Total wall time     : {total_elapsed / 3600:.2f} h  ({total_elapsed:.0f} s)",
            f"   Avg per episode     : {total_elapsed / n / 60:.1f} min",
        ]
    lines.append("=" * 50)
    metrics_path = run_dir / "metrics.log"
    with open(metrics_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Base output directory. A timestamped run_YYYYMMDD_HHMMSS/ subfolder is created inside.",
    )
    parser.add_argument(
        "--episodes", default=None,
        help="Episodes to process: 'all' (default), a range '0-9', or comma-separated '0,5,10'.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Process the first N episodes (episodes 0 to N-1). Ignored when --episodes is set.",
    )
    parser.add_argument(
        "--full-res", action="store_true",
        help=(
            "Horizontal stack (1280×480) instead of vertical (640×960). "
            "Higher per-view detail but ~33%% more VRAM (~86–97 GB). "
            "May OOM — reduce 'resolution' in inference_config.json to '480' if needed."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run episodes whose top.mp4 and wrist.mp4 outputs already exist.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to inference_config.json. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--prompt", type=Path, default=DEFAULT_PROMPT_PATH,
        help=f"Path to style_prompt.txt containing {{cell_position}}. Default: {DEFAULT_PROMPT_PATH}",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT,
        help=f"Root directory for control video cache. Default: {DEFAULT_CACHE_ROOT}",
    )
    # Inference overrides — supersede values in inference_config.json
    parser.add_argument("--num-steps",  type=int,   default=None, help="Override num_steps.")
    parser.add_argument("--guidance",   type=int,   default=None, help="Override guidance (0–7).")
    parser.add_argument("--seed",       type=int,   default=None, help="Override seed.")
    parser.add_argument(
        "--image-context-path", type=Path, default=None,
        help="Override image_context_path. Reference image for style transfer. "
             "Automatically disables vis control (they conflict).",
    )

    args = parser.parse_args()

    # ── Episode selection ──
    try:
        episodes = parse_episodes(args.episodes, args.num_episodes)
    except ValueError as e:
        parser.error(str(e))

    if not episodes:
        parser.error("No episodes selected.")
    for ep in episodes:
        if ep < 0 or ep >= NUM_EPISODES:
            parser.error(f"Episode {ep} is out of range [0, {NUM_EPISODES - 1}].")

    # ── Load config and apply CLI overrides ──
    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    if args.num_steps           is not None: config["num_steps"]           = args.num_steps
    if args.guidance            is not None: config["guidance"]            = args.guidance
    if args.seed                is not None: config["seed"]                = args.seed
    if args.image_context_path  is not None: config["image_context_path"]  = str(args.image_context_path.resolve())

    # ── Resolve and validate image_context_path ──
    image_ctx = config.get("image_context_path")
    if image_ctx and not Path(image_ctx).exists():
        parser.error(f"image_context_path does not exist: {image_ctx}")

    # ── Active controls — drop vis if image_context_path is set (they conflict) ──
    active_controls = [k for k in CONTROL_KEYS if config.get(k) is not None]
    if image_ctx and "vis" in active_controls:
        print("WARNING: image_context_path is set — vis control disabled (mutually exclusive).")
        active_controls.remove("vis")

    if not active_controls:
        parser.error("No active controls in config. Set at least one of: edge, depth, seg, vis.")

    # ── Load prompt template ──
    prompt_path = args.prompt.resolve()
    if not prompt_path.exists():
        parser.error(f"Prompt file not found: {prompt_path}")
    prompt_template = prompt_path.read_text().strip()
    if "{cell_position}" not in prompt_template:
        print("WARNING: prompt file has no {cell_position} placeholder — all episodes get the same prompt.")

    # ── Stacking mode ──
    mode_key     = "hstack" if args.full_res else "vstack"
    mode_label   = "hstack 1280×480 (full-res)" if args.full_res else "vstack 640×960 (safe)"
    cache_root   = args.cache_dir.resolve()

    # ── Create timestamped run directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = args.output_dir.resolve() / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir   = run_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nRun directory : {run_dir}")
    print(f"Mode          : {mode_label}")
    print(f"Episodes      : {len(episodes)}  ({episodes[0]}–{episodes[-1]})" if len(episodes) > 1 else f"Episodes      : 1  (episode {episodes[0]})")
    print(f"Controls      : {active_controls}")
    print(f"num_steps     : {config.get('num_steps', 35)}")
    print(f"seed          : {config.get('seed', 2025)}")
    if image_ctx:
        print(f"Image ref     : {image_ctx}")
    print(f"Cache root    : {cache_root}\n")

    if args.full_res:
        print(
            "VRAM WARNING: full-res mode processes at 1280×704 internally (~33% more latent tokens "
            "than single-view). Estimated peak VRAM ~86–97 GB on RTX 6000 96 GB. If OOM, omit "
            "--full-res or set \"resolution\": \"480\" in inference_config.json.\n"
        )

    # ── Initialise inference pipeline (loads model weights once) ──
    from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
    init_environment()

    from cosmos_transfer2.config import InferenceArguments, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference

    setup_args = SetupArguments(output_dir=run_dir, disable_guardrails=True, model="edge")
    init_output_dir(run_dir, profile=False)

    # batch_hint_keys must be sorted in CONTROL_KEYS order so the right model checkpoint loads.
    batch_hint_keys = sorted(active_controls, key=lambda k: CONTROL_KEYS.index(k))
    inference = Control2WorldInference(setup_args, batch_hint_keys=batch_hint_keys)

    # ── Process episodes ──────────────────────────────────────────────────────
    t_run_start = time.perf_counter()
    processed: list[int] = []

    for i, ep_idx in enumerate(episodes):
        ep_name  = f"episode_{ep_idx:06d}"
        ep_dir   = run_dir / ep_name
        top_out  = ep_dir / "top.mp4"
        wrist_out = ep_dir / "wrist.mp4"

        print(f"[{i + 1}/{len(episodes)}] {ep_name}  —  {get_cell_position(ep_idx)}")

        # Skip if already completed
        if not args.force and top_out.exists() and wrist_out.exists():
            print(f"         Skipping (outputs exist). Use --force to re-run.\n")
            continue

        top_src   = TOP_DIR   / f"{ep_name}.mp4"
        wrist_src = WRIST_DIR / f"{ep_name}.mp4"
        if not top_src.exists() or not wrist_src.exists():
            print(f"         WARNING: missing input video(s) — skipping.\n")
            continue

        cell   = get_cell_position(ep_idx)
        prompt = prompt_template.replace("{cell_position}", cell)

        # 1. Check cache for pre-computed control videos
        cached_controls = get_cached_controls(ep_idx, cache_root, mode_key, active_controls)
        if cached_controls:
            print(f"         Cache hit : {list(cached_controls)}")
        missing = [c for c in active_controls if c not in cached_controls]
        if missing:
            print(f"         Generating: {missing}")

        # 2. Create combined video (ffmpeg)
        combined_path = tmp_dir / f"{ep_name}_combined.mp4"
        create_combined_video(top_src.resolve(), wrist_src.resolve(), combined_path, args.full_res)

        # 3. Build and validate InferenceArguments
        sample_dict = build_sample_dict(
            ep_idx=ep_idx,
            combined_path=combined_path,
            prompt=prompt,
            config=config,
            cached_controls=cached_controls,
            active_controls=active_controls,
        )
        sample = InferenceArguments.model_validate(sample_dict)

        # 4. Run inference (metrics written to run_dir/metrics.log by the pipeline)
        inference.generate([sample], run_dir)

        # 5. Organise outputs into per-episode subdirectory
        ep_dir.mkdir(exist_ok=True)
        sample_name = sample_dict["name"]

        combined_generated = run_dir / f"{sample_name}.mp4"
        if combined_generated.exists():
            shutil.move(str(combined_generated), str(ep_dir / "combined.mp4"))

        for ctrl in active_controls:
            ctrl_src_path = run_dir / f"{sample_name}_control_{ctrl}.mp4"
            if ctrl_src_path.exists():
                shutil.move(str(ctrl_src_path), str(ep_dir / f"control_{ctrl}.mp4"))

        for ext in (".json", ".txt"):
            sidecar = run_dir / f"{sample_name}{ext}"
            if sidecar.exists():
                shutil.move(str(sidecar), str(ep_dir / f"combined{ext}"))

        # 6. Split combined → top.mp4 + wrist.mp4
        combined_dst = ep_dir / "combined.mp4"
        if combined_dst.exists():
            split_combined_video(combined_dst, top_out, wrist_out, args.full_res)
            print(f"         Split    : top.mp4 + wrist.mp4")

        # 7. Cache freshly generated control videos
        save_controls_to_cache(
            ep_dir=ep_dir,
            ep_idx=ep_idx,
            cache_root=cache_root,
            mode_key=mode_key,
            active_controls=active_controls,
            already_cached=cached_controls,
        )

        # 8. Delete temporary combined input video
        if combined_path.exists():
            combined_path.unlink()

        processed.append(ep_idx)
        print(f"         Output   : {ep_dir.relative_to(run_dir.parent)}\n")

    # ── Run summary ───────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_run_start
    write_run_summary(run_dir, processed, total_elapsed)

    cleanup_environment()

    n = len(processed)
    skipped = len(episodes) - n
    print(f"Done. {n} episode(s) completed"
          + (f", {skipped} skipped" if skipped else "")
          + f" in {total_elapsed / 3600:.2f} h.")
    print(f"Outputs : {run_dir}")
    print(f"Metrics : {run_dir / 'metrics.log'}")


if __name__ == "__main__":
    main()
