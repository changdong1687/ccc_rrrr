#!/usr/bin/env python3
"""Convert LIBERO HDF5 demonstrations to LeRobot v2 + GEAR/DreamZero format."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger("convert_libero_data")


CAMERA_KEYS = {
    "agentview": ("agentview_rgb", "agentview_image", "agentview"),
    "eye_in_hand": ("eye_in_hand_rgb", "robot0_eye_in_hand_image", "eye_in_hand"),
}
JOINT_KEYS = ("robot0_joint_pos", "joint_pos", "joint_position", "joint_states")
GRIPPER_KEYS = ("robot0_gripper_qpos", "robot0_gripper_pos", "gripper_qpos", "gripper_position", "gripper_states")


def _read_attr_text(obj: h5py.Group, key: str) -> str | None:
    value = obj.attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def _find_first(group: h5py.Group, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in group:
            return name
    return None


def _ensure_rgb(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape (T,H,W,C), got {frames.shape}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.dtype != np.uint8:
        if frames.max(initial=0) <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps, codec="libx264", quality=8)
        return
    except Exception as imageio_error:
        log.debug("imageio video write failed for %s: %s", path, imageio_error)

    import cv2

    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _iter_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    patterns = ("*.hdf5", "*.h5")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_path.rglob(pattern)))
    return files


def _demo_groups(h5: h5py.File) -> list[tuple[str, h5py.Group]]:
    root = h5["data"] if "data" in h5 else h5
    demos = [(name, root[name]) for name in sorted(root.keys()) if isinstance(root[name], h5py.Group)]
    return [(name, group) for name, group in demos if "obs" in group]


def _task_text(h5: h5py.File, demo: h5py.Group, fallback: str) -> str:
    for owner in (demo, h5["data"] if "data" in h5 else h5, h5):
        for key in ("language", "task", "task_description", "problem", "bddl_file_name"):
            value = _read_attr_text(owner, key)
            if value:
                return value
    return fallback.replace("_", " ")


def _load_demo(
    h5: h5py.File,
    demo_name: str,
    demo: h5py.Group,
    task: str,
    fps: int,
    action_source: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int]]:
    obs = demo["obs"]
    joint_key = _find_first(obs, JOINT_KEYS)
    gripper_key = _find_first(obs, GRIPPER_KEYS)
    if joint_key is None or gripper_key is None:
        raise KeyError(f"{demo_name}: could not find joint/gripper state keys under obs/")

    joint = np.asarray(obs[joint_key], dtype=np.float32)
    gripper = np.asarray(obs[gripper_key], dtype=np.float32)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    gripper = gripper[:, :1]
    horizon = min(len(joint), len(gripper))

    videos = {}
    for view, candidates in CAMERA_KEYS.items():
        key = _find_first(obs, candidates)
        if key is None:
            raise KeyError(f"{demo_name}: missing camera for {view}; tried {candidates}")
        frames = _ensure_rgb(obs[key][:horizon])
        videos[view] = frames
        horizon = min(horizon, len(frames))

    joint = joint[:horizon]
    gripper = gripper[:horizon]
    state = np.concatenate([joint, gripper], axis=-1).astype(np.float32)

    if action_source == "next_state":
        action_joint = np.concatenate([joint[1:], joint[-1:]], axis=0)
        action_gripper = np.concatenate([gripper[1:], gripper[-1:]], axis=0)
        action = np.concatenate([action_joint, action_gripper], axis=-1).astype(np.float32)
    else:
        if "actions" not in demo:
            raise KeyError(f"{demo_name}: --action-source=hdf5_actions but no actions dataset found")
        raw_action = np.asarray(demo["actions"][:horizon], dtype=np.float32)
        if raw_action.ndim == 1:
            raw_action = raw_action[:, None]
        if raw_action.shape[-1] >= joint.shape[-1] + gripper.shape[-1]:
            action_joint = raw_action[:, : joint.shape[-1]]
            action_gripper = raw_action[:, joint.shape[-1] : joint.shape[-1] + gripper.shape[-1]]
        elif raw_action.shape[-1] > 1:
            action_joint = raw_action[:, :-1]
            action_gripper = raw_action[:, -1:]
        else:
            action_joint = raw_action
            action_gripper = np.zeros((horizon, 1), dtype=np.float32)
        action = np.concatenate([action_joint, action_gripper], axis=-1).astype(np.float32)

    timestamps = np.arange(horizon, dtype=np.float64) / float(fps)
    rows = {
        "observation.state": list(state),
        "action": list(action),
        "timestamp": timestamps,
        "frame_index": np.arange(horizon, dtype=np.int64),
        "episode_index": np.zeros(horizon, dtype=np.int64),
        "task_index": np.zeros(horizon, dtype=np.int64),
        "annotation.task": [task] * horizon,
    }
    dims = {
        "joint_dim": int(joint.shape[-1]),
        "gripper_dim": int(gripper.shape[-1]),
        "action_joint_dim": int(action_joint.shape[-1]),
        "action_gripper_dim": int(action_gripper.shape[-1]),
    }
    return pd.DataFrame(rows), videos, dims


def _stats(parquet_paths: list[Path], columns: list[str]) -> dict:
    stats = {}
    for column in columns:
        arrays = []
        for path in parquet_paths:
            df = pd.read_parquet(path)
            arrays.append(np.stack(df[column].values).astype(np.float64))
        data = np.concatenate(arrays, axis=0)
        stats[column] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def _relative_stats(parquet_paths: list[Path], joint_dim: int, action_joint_dim: int, action_horizon: int) -> dict:
    if joint_dim != action_joint_dim:
        log.warning(
            "Skipping relative joint stats because state joint dim (%d) != action joint dim (%d).",
            joint_dim,
            action_joint_dim,
        )
        return {}
    values = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        state = np.stack(df["observation.state"].values).astype(np.float64)[:, :joint_dim]
        action = np.stack(df["action"].values).astype(np.float64)[:, :action_joint_dim]
        for i in range(max(len(df) - action_horizon, 0)):
            values.append(action[i : min(i + action_horizon, len(df))] - state[i])
    if not values:
        return {}
    data = np.concatenate(values, axis=0)
    return {
        "joint_position": {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to overwrite")
        shutil.rmtree(output)
    (output / "meta").mkdir(parents=True)

    hdf5_files = _iter_hdf5_files(input_path)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {input_path}")

    task_to_index: dict[str, int] = {}
    episodes: list[dict] = []
    parquet_paths: list[Path] = []
    first_dims: dict[str, int] | None = None
    first_video_shape: dict[str, tuple[int, int, int]] = {}
    episode_index = 0

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as h5:
            fallback_task = hdf5_path.stem
            for demo_name, demo in tqdm(_demo_groups(h5), desc=hdf5_path.name):
                task = args.task or _task_text(h5, demo, fallback_task)
                task_index = task_to_index.setdefault(task, len(task_to_index))
                df, videos, dims = _load_demo(h5, demo_name, demo, task, args.fps, args.action_source)
                if len(df) < args.min_episode_len:
                    continue

                df["episode_index"] = episode_index
                df["task_index"] = task_index
                chunk = episode_index // args.chunk_size
                data_path = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
                data_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(data_path)
                parquet_paths.append(data_path)

                for view, frames in videos.items():
                    video_path = (
                        output
                        / "videos"
                        / f"chunk-{chunk:03d}"
                        / f"observation.images.{view}"
                        / f"episode_{episode_index:06d}.mp4"
                    )
                    _write_video(video_path, frames[: len(df)], args.fps)
                    first_video_shape.setdefault(view, frames.shape[1:])

                if first_dims is None:
                    first_dims = dims
                episodes.append({"episode_index": episode_index, "tasks": [task], "length": len(df)})
                episode_index += 1

    if episode_index == 0 or first_dims is None:
        raise RuntimeError("No usable LIBERO episodes were converted.")

    tasks = [{"task_index": index, "task": task} for task, index in sorted(task_to_index.items(), key=lambda item: item[1])]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "libero_sim",
        "total_episodes": episode_index,
        "total_frames": int(sum(ep["length"] for ep in episodes)),
        "total_tasks": len(tasks),
        "total_videos": episode_index * len(CAMERA_KEYS),
        "total_chunks": int((episode_index + args.chunk_size - 1) // args.chunk_size),
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [first_dims["joint_dim"] + first_dims["gripper_dim"]],
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": [first_dims["action_joint_dim"] + first_dims["action_gripper_dim"]],
                "names": ["action"],
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.task": {"dtype": "string", "shape": [1], "names": None},
        },
    }
    for view, shape in first_video_shape.items():
        height, width, channels = shape
        info["features"][f"observation.images.{view}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": args.fps},
        }

    meta = output / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "embodiment.json").write_text(json.dumps({"robot_type": "libero_sim", "embodiment_tag": "libero_sim"}, indent=4))
    modality = {
        "state": {
            "joint_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": first_dims["joint_dim"],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
            "gripper_position": {
                "original_key": "observation.state",
                "start": first_dims["joint_dim"],
                "end": first_dims["joint_dim"] + first_dims["gripper_dim"],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
        },
        "action": {
            "joint_position": {
                "original_key": "action",
                "start": 0,
                "end": first_dims["action_joint_dim"],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
            "gripper_position": {
                "original_key": "action",
                "start": first_dims["action_joint_dim"],
                "end": first_dims["action_joint_dim"] + first_dims["action_gripper_dim"],
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
        },
        "video": {
            "agentview": {"original_key": "observation.images.agentview"},
            "eye_in_hand": {"original_key": "observation.images.eye_in_hand"},
        },
        "annotation": {"task": {"original_key": "annotation.task"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))
    (meta / "stats.json").write_text(json.dumps(_stats(parquet_paths, ["observation.state", "action"]), indent=4))
    rel_stats = _relative_stats(parquet_paths, first_dims["joint_dim"], first_dims["action_joint_dim"], args.action_horizon)
    if rel_stats:
        (meta / "relative_stats_dreamzero.json").write_text(json.dumps(rel_stats, indent=4))
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "episodes.jsonl", episodes)

    print(f"Converted {episode_index} episodes to {output}")
    print(f"Tasks: {len(tasks)} | FPS: {args.fps} | action_source={args.action_source}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="LIBERO HDF5 file or directory")
    parser.add_argument("--output", required=True, help="Output LeRobot v2 + GEAR dataset directory")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--action-source", choices=["next_state", "hdf5_actions"], default="next_state")
    parser.add_argument("--task", default=None, help="Override language task for every episode")
    parser.add_argument("--min-episode-len", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")
    convert(args)


if __name__ == "__main__":
    main()
