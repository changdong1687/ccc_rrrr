#!/usr/bin/env python3
"""Offline overfit sanity check for the LIBERO joint-action policy.

This does NOT run the simulator. It replays the *recorded* observations from a
converted LIBERO dataset (the same data used for training) through the exact
inference path used at eval time (``LiberoJointPolicy`` from
``eval_libero_server.py``), and compares the predicted actions against the
ground-truth actions stored in the dataset.

Why: it decouples two questions that the in-sim rollout conflates:
  * Did the model actually learn the action mapping?  -> prediction vs GT error.
  * Does the sim controller / gripper execute it correctly? -> separate concern.

If the model is overfit on a single task and predictions still don't match the
GT actions here, the problem is in training/data, not in the controller setup.

Usage (run from repo root):

    python simulation/libero/debug/eval_overfit_offline.py \
        --model-path ./checkpoints/dreamzero_libero_overfit_debug/checkpoint-5000 \
        --dataset-root ./data/libero_debug_onetask \
        --episode-index 0 \
        --output ./runs/overfit_debug
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_server_module():
    """Import eval_libero_server.py by path to reuse LiberoJointPolicy."""
    server_path = Path(__file__).resolve().parents[1] / "eval_libero_server.py"
    spec = importlib.util.spec_from_file_location("eval_libero_server", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_dist() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _read_video_frames(path: Path) -> np.ndarray:
    """Return all frames of an mp4 as a (T, H, W, 3) uint8 array."""
    try:
        import decord

        reader = decord.VideoReader(str(path))
        frames = reader[:].asnumpy()
        return np.asarray(frames, dtype=np.uint8)
    except Exception:
        pass
    import imageio.v3 as iio

    return np.asarray(iio.imread(path, plugin="pyav"), dtype=np.uint8)


def _episode_paths(dataset_root: Path, episode_index: int) -> tuple[Path, dict[str, Path]]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    data_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    video_keys = [k for k in info["features"] if k.startswith("observation.images.")]
    video_paths = {
        key: dataset_root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        for key in video_keys
    }
    return data_path, video_paths


def _view_name(video_key: str) -> str:
    # "observation.images.agentview" -> "agentview"
    return video_key.split(".")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-eval-steps", type=int, default=None, help="Cap on frames to evaluate (default: all).")
    parser.add_argument("--max-chunk-size", type=int, default=None)
    parser.add_argument("--num-frame-per-block", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("./runs/overfit_debug"))
    args = parser.parse_args()

    _ensure_dist()
    server = _load_server_module()

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    model_config_overrides: list[str] = []
    if args.max_chunk_size is not None:
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={args.max_chunk_size}"
        )
    if args.num_frame_per_block is not None:
        model_config_overrides.append(f"action_head_cfg.config.num_frame_per_block={args.num_frame_per_block}")
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.num_frame_per_block={args.num_frame_per_block}"
        )

    print(f"Loading policy from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device=args.device,
        model_config_overrides=model_config_overrides,
    )
    wrapper = server.LiberoJointPolicy(policy)

    # --- Load the recorded episode (same data used for training) ---
    data_path, video_paths = _episode_paths(args.dataset_root, args.episode_index)
    print(f"Reading {data_path}")
    df = pd.read_parquet(data_path)
    state = np.stack(df["observation.state"].values).astype(np.float64)  # [T, joint+gripper]
    gt_action = np.stack(df["action"].values).astype(np.float64)  # [T, joint+gripper]
    prompt = str(df["annotation.task"].values[0])
    horizon = len(df)

    videos = {_view_name(k): _read_video_frames(p) for k, p in video_paths.items()}
    for name, frames in videos.items():
        print(f"video[{name}] = {frames.shape}")

    # Infer joint/gripper split from the action dims via modality.json.
    modality = json.loads((args.dataset_root / "meta" / "modality.json").read_text())
    n_joint = int(modality["action"]["joint_position"]["end"] - modality["action"]["joint_position"]["start"])
    print(f"prompt={prompt!r}  horizon={horizon}  n_joint={n_joint}")

    n_steps = horizon if args.max_eval_steps is None else min(horizon, args.max_eval_steps)

    wrapper.reset({"session_id": "overfit_debug"})

    pred0 = np.zeros((n_steps, gt_action.shape[1]), dtype=np.float64)  # first predicted step
    gt0 = gt_action[:n_steps].copy()
    chunk_mse = []  # full-chunk MSE vs GT[t:t+N]

    for t in range(n_steps):
        obs: dict[str, Any] = {
            "observation/agentview": videos["agentview"][min(t, len(videos["agentview"]) - 1)],
            "observation/eye_in_hand": videos["eye_in_hand"][min(t, len(videos["eye_in_hand"]) - 1)],
            "observation/joint_position": state[t, :n_joint],
            "observation/gripper_position": state[t, n_joint:],
            "prompt": prompt,
            "session_id": "overfit_debug",
        }
        result = wrapper.infer(obs)
        pred_chunk = np.asarray(result["actions"], dtype=np.float64)  # [N, joint+gripper]
        pred0[t] = pred_chunk[0]

        # Compare the full predicted chunk against the GT actions that follow.
        n = min(len(pred_chunk), horizon - t)
        if n > 0:
            chunk_mse.append(float(np.mean((pred_chunk[:n] - gt_action[t : t + n]) ** 2)))

        if t % 20 == 0:
            j_err = np.abs(pred_chunk[0, :n_joint] - gt_action[t, :n_joint]).mean()
            g_pred = pred_chunk[0, n_joint] if pred_chunk.shape[1] > n_joint else float("nan")
            g_gt = gt_action[t, n_joint] if gt_action.shape[1] > n_joint else float("nan")
            print(f"[t={t:3d}] joint_MAE={j_err:.4f}  gripper pred={g_pred:.4f} gt={g_gt:.4f}")

    # --- Metrics ---
    joint_abs_err = np.abs(pred0[:, :n_joint] - gt0[:, :n_joint])  # [T, n_joint]
    gripper_pred = pred0[:, n_joint] if pred0.shape[1] > n_joint else None
    gripper_gt = gt0[:, n_joint] if gt0.shape[1] > n_joint else None

    summary = {
        "model_path": args.model_path,
        "dataset_root": str(args.dataset_root),
        "episode_index": args.episode_index,
        "prompt": prompt,
        "n_steps": n_steps,
        "joint_mae_overall": float(joint_abs_err.mean()),
        "joint_mae_per_dim": joint_abs_err.mean(axis=0).tolist(),
        "joint_max_abs_err": float(joint_abs_err.max()),
        "chunk_mse_mean": float(np.mean(chunk_mse)) if chunk_mse else None,
    }
    if gripper_pred is not None:
        summary["gripper_mae"] = float(np.abs(gripper_pred - gripper_gt).mean())
        summary["gripper_pred_min"] = float(gripper_pred.min())
        summary["gripper_pred_max"] = float(gripper_pred.max())
        summary["gripper_gt_min"] = float(gripper_gt.min())
        summary["gripper_gt_max"] = float(gripper_gt.max())

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overfit_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        args.output / "overfit_pred_vs_gt.npz",
        pred0=pred0,
        gt0=gt0,
        n_joint=n_joint,
    )

    print("\n===== OVERFIT SANITY SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved arrays to {args.output / 'overfit_pred_vs_gt.npz'}")
    print(
        "\nInterpretation:\n"
        "  joint_mae_overall (rad): small (e.g. < ~0.02) => model reproduces training actions => learning works.\n"
        "  large                                         => training/data problem, not the controller.\n"
        "  gripper pred range vs gt range: use to calibrate --gripper-threshold for the sim eval."
    )


if __name__ == "__main__":
    main()
