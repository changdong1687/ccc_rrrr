#!/usr/bin/env python3
"""Closed-loop DreamZero evaluation in LIBERO simulation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


AGENTVIEW_KEYS = ("agentview_image", "agentview_rgb", "agentview")
EYE_IN_HAND_KEYS = ("robot0_eye_in_hand_image", "eye_in_hand_rgb", "eye_in_hand_image", "eye_in_hand")
JOINT_KEYS = ("robot0_joint_pos", "joint_pos", "joint_position")
GRIPPER_KEYS = ("robot0_gripper_qpos", "robot0_gripper_pos", "gripper_qpos", "gripper_position")


def _first(obs: dict, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in obs:
            return obs[name]
    raise KeyError(f"None of {names} found in observation keys: {sorted(obs.keys())}")


def _rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _state(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    joint = np.asarray(_first(obs, JOINT_KEYS), dtype=np.float64).reshape(1, -1)
    gripper = np.asarray(_first(obs, GRIPPER_KEYS), dtype=np.float64).reshape(1, -1)
    return joint, gripper


def _task_language(task: Any) -> str:
    for attr in ("language", "task", "description", "name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    if isinstance(task, dict):
        for key in ("language", "task", "description", "name"):
            if task.get(key):
                return str(task[key])
    return str(task).replace("_", " ")


def _load_benchmark(name: str):
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise ImportError("Install LIBERO first: pip install -e /path/to/LIBERO") from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if name not in benchmark_dict:
        raise KeyError(f"Unknown LIBERO benchmark {name!r}. Available: {sorted(benchmark_dict)}")
    return benchmark_dict[name]()


def _make_env(task: Any, bddl_root: str | None, seed: int):
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError("Could not import LIBERO OffScreenRenderEnv.") from exc

    bddl_file = getattr(task, "bddl_file", None) or getattr(task, "bddl_file_name", None)
    if bddl_file and bddl_root and not os.path.isabs(str(bddl_file)):
        bddl_file = str(Path(bddl_root) / str(bddl_file))
    init_states = getattr(task, "init_states", None)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env, init_states


def _reset_env(env, init_states, episode_idx: int):
    obs = env.reset()
    if init_states is not None and len(init_states) > 0:
        state = init_states[episode_idx % len(init_states)]
        if hasattr(env, "set_init_state"):
            obs = env.set_init_state(state)
        elif hasattr(env, "sim") and hasattr(env.sim, "set_state_from_flattened"):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obs = env._get_observations()
    return obs


def _prediction_to_env_action(result: Batch, env_action_dim: int) -> np.ndarray:
    act = result.act
    joint = act.get("action.joint_position", None)
    gripper = act.get("action.gripper_position", None)
    if joint is None:
        raise KeyError(f"Policy output has no action.joint_position. Keys: {list(act.keys())}")

    if isinstance(joint, torch.Tensor):
        joint = joint.detach().cpu().numpy()
    if isinstance(gripper, torch.Tensor):
        gripper = gripper.detach().cpu().numpy()
    joint = np.asarray(joint)[0].reshape(-1)
    parts = [joint]
    if gripper is not None:
        parts.append(np.asarray(gripper)[0].reshape(-1))
    action = np.concatenate(parts).astype(np.float64)
    if len(action) < env_action_dim:
        action = np.pad(action, (0, env_action_dim - len(action)))
    return action[:env_action_dim]


def build_obs(obs: dict, prompt: str) -> dict:
    joint, gripper = _state(obs)
    return {
        "video.agentview": _rgb(_first(obs, AGENTVIEW_KEYS))[None],
        "video.eye_in_hand": _rgb(_first(obs, EYE_IN_HAND_KEYS))[None],
        "state.joint_position": joint,
        "state.gripper_position": gripper,
        "annotation.task": prompt,
    }


def evaluate(args: argparse.Namespace) -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device=args.device,
    )
    benchmark = _load_benchmark(args.benchmark)
    num_tasks = benchmark.n_tasks if args.num_tasks is None else min(args.num_tasks, benchmark.n_tasks)

    results = []
    for task_id in range(num_tasks):
        task = benchmark.get_task(task_id)
        language = args.prompt or _task_language(task)
        env, init_states = _make_env(task, args.bddl_root, args.seed + task_id)
        task_successes = 0

        for episode_idx in range(args.episodes_per_task):
            obs = _reset_env(env, init_states, episode_idx)
            success = False
            start = time.perf_counter()
            for step in range(args.max_steps):
                with torch.inference_mode():
                    result, _ = policy.lazy_joint_forward_causal(Batch(obs=build_obs(obs, language)))
                action = _prediction_to_env_action(result, env.action_spec[0].shape[0])
                obs, reward, done, info = env.step(action)
                success = bool(info.get("success", False) or reward > 0.5)
                if args.render and hasattr(env, "render"):
                    env.render()
                if success or done:
                    break
            elapsed = time.perf_counter() - start
            task_successes += int(success)
            row = {
                "task_id": task_id,
                "episode": episode_idx,
                "success": success,
                "steps": step + 1,
                "seconds": elapsed,
                "language": language,
            }
            results.append(row)
            print(
                f"task={task_id:02d} episode={episode_idx:02d} "
                f"success={success} steps={step + 1} language={language!r}"
            )

        env.close()
        rate = task_successes / float(args.episodes_per_task)
        print(f"Task {task_id:02d} success rate: {rate:.3f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text("\n".join(json.dumps(row) for row in results) + "\n")
    overall = float(np.mean([row["success"] for row in results])) if results else 0.0
    summary = {"benchmark": args.benchmark, "success_rate": overall, "episodes": len(results)}
    (output / "summary.json").write_text(json.dumps(summary, indent=4))
    print(f"Overall success rate: {overall:.3f}. Results saved to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, help="DreamZero checkpoint directory")
    parser.add_argument("--benchmark", default="libero_spatial", help="LIBERO benchmark name")
    parser.add_argument("--bddl-root", default=None, help="Optional root for relative BDDL files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=None, help="Override task language")
    parser.add_argument("--output-dir", default="results_libero")
    parser.add_argument("--render", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
