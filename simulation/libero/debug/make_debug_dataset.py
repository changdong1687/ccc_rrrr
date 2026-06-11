#!/usr/bin/env python3
"""Subset an already-converted LIBERO LeRobot+GEAR dataset down to a few episodes.

Use this to build the strongest possible overfit case (e.g. a single episode)
from an existing converted dataset, without re-running the HDF5 converter.

It copies the selected episodes' parquet + videos verbatim and rewrites only
meta/episodes.jsonl and the totals in meta/info.json. All other meta files
(modality.json, stats.json, relative_stats_dreamzero.json, tasks.jsonl) are
copied unchanged, so normalization stays identical to the source dataset.

Usage:
    python simulation/libero/debug/make_debug_dataset.py \
        --src ./data/libero_spatial \
        --dst ./data/libero_debug_onetask \
        --num-episodes 1 --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Existing converted dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Output subset dataset root.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Keep the first N episodes.")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit episode indices to keep (overrides --num-episodes).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        if not args.force:
            raise FileExistsError(f"{dst} exists; pass --force")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    video_keys = [k for k in info["features"] if k.startswith("observation.images.")]

    episodes = _load_jsonl(src / "meta" / "episodes.jsonl")
    by_index = {ep["episode_index"]: ep for ep in episodes}

    if args.episode_indices is not None:
        keep_indices = args.episode_indices
    else:
        keep_indices = [ep["episode_index"] for ep in episodes[: args.num_episodes]]
    missing = [i for i in keep_indices if i not in by_index]
    if missing:
        raise KeyError(f"episode indices not found in source: {missing}")

    (dst / "meta").mkdir(parents=True)

    # Copy all meta files verbatim; episodes.jsonl + info.json are rewritten below.
    for name in ("modality.json", "stats.json", "relative_stats_dreamzero.json",
                 "relative_horizon_stats_dreamzero.json", "tasks.jsonl", "embodiment.json"):
        srcf = src / "meta" / name
        if srcf.exists():
            shutil.copy2(srcf, dst / "meta" / name)

    kept_episodes = []
    total_frames = 0
    for idx in keep_indices:
        ep = by_index[idx]
        kept_episodes.append(ep)
        total_frames += int(ep["length"])
        chunk = idx // chunk_size

        # parquet
        rel_parquet = Path("data") / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
        (dst / rel_parquet).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel_parquet, dst / rel_parquet)

        # videos (one per camera view)
        for key in video_keys:
            rel_video = Path("videos") / f"chunk-{chunk:03d}" / key / f"episode_{idx:06d}.mp4"
            (dst / rel_video).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel_video, dst / rel_video)

    _write_jsonl(dst / "meta" / "episodes.jsonl", kept_episodes)

    info["total_episodes"] = len(kept_episodes)
    info["total_frames"] = total_frames
    info["total_videos"] = len(kept_episodes) * len(video_keys)
    max_chunk = max((i // chunk_size for i in keep_indices), default=0)
    info["total_chunks"] = max_chunk + 1
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"Wrote {len(kept_episodes)} episode(s) ({total_frames} frames) to {dst}")
    print(f"Kept episode indices: {keep_indices}")


if __name__ == "__main__":
    main()
