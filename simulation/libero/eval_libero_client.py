#!/usr/bin/env python3
"""Run LIBERO benchmark rollouts against a DreamZero joint-action websocket server."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import uuid
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import websockets.sync.client
from tqdm.auto import tqdm

DEFAULT_LIBERO_ROOT = Path(__file__).resolve().parents[2] / "LIBERO"
AGENTVIEW_KEYS = ("agentview_image", "agentview_rgb", "agentview")
EYE_IN_HAND_KEYS = ("robot0_eye_in_hand_image", "eye_in_hand_rgb", "eye_in_hand_image", "eye_in_hand")
JOINT_KEYS = ("robot0_joint_pos", "joint_pos", "joint_position")
GRIPPER_KEYS = ("robot0_gripper_qpos", "robot0_gripper_pos", "gripper_qpos", "gripper_position")


def first(obs: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in obs:
            return obs[name]
    raise KeyError(f"None of {names} found in observation keys: {sorted(obs.keys())}")


def rgb(image: Any) -> np.ndarray:
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


def state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    joint = np.asarray(first(obs, JOINT_KEYS), dtype=np.float64).reshape(-1)
    gripper = np.asarray(first(obs, GRIPPER_KEYS), dtype=np.float64).reshape(-1)[:1]
    return joint, gripper


def make_gripper_command(value: Any, threshold: float, invert: bool) -> float:
    """Map a predicted gripper joint position to a robosuite gripper command.

    The policy is trained to predict the next gripper *qpos* (a continuous finger
    position, larger ~= more open for the Panda gripper), but the simulator's
    gripper sub-controller only looks at the *sign* of the action (+1 closes,
    -1 opens). We therefore binarize the predicted position around ``threshold``.
    Use ``invert`` if the open/close sense is flipped for your gripper.
    """
    closed = float(np.asarray(value).reshape(-1)[0]) <= threshold
    command = 1.0 if closed else -1.0
    return -command if invert else command


def build_joint_position_controller_config(
    controller: str, joint_delta_bound: float, joint_kp: float | None
):
    """Build a robosuite controller config with identity action scaling.

    Identity scaling (input range == output range) lets us command joint *deltas*
    in radians directly, instead of robosuite's default tiny 0.05 rad/step range.
    Returns ``None`` if robosuite is unavailable so the caller can fall back to
    LIBERO's controller-name API.
    """
    try:
        from robosuite.controllers import load_controller_config
    except Exception:  # pragma: no cover - depends on eval env
        return None
    cfg = load_controller_config(default_controller=controller)
    if controller == "JOINT_POSITION":
        cfg["input_max"] = joint_delta_bound
        cfg["input_min"] = -joint_delta_bound
        cfg["output_max"] = joint_delta_bound
        cfg["output_min"] = -joint_delta_bound
        if joint_kp is not None:
            cfg["kp"] = joint_kp
    return cfg


def _override_controller_scaling(env: Any, joint_delta_bound: float, joint_kp: float | None) -> None:
    """Force identity action scaling on an already-constructed LIBERO env.

    Used when LIBERO only accepts a controller *name* (so the config cannot be
    passed in). We reach the live controller object and reset its scaling.
    """
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", None)
    if not robots:
        print(
            "[warn] Could not locate robots to override JOINT_POSITION scaling; "
            "the controller may keep its default 0.05 rad/step range and undershoot targets."
        )
        return
    overridden = False
    for robot in robots:
        ctrl = getattr(robot, "controller", None) or getattr(robot, "_controller", None)
        if ctrl is None:
            continue
        overridden = True
        # robosuite's control_limits / scale_action expect per-joint *arrays*,
        # not scalars, so size everything to the controller's action dim.
        dim = getattr(ctrl, "control_dim", None)
        if not dim:
            qpos_index = getattr(ctrl, "qpos_index", None)
            dim = len(qpos_index) if qpos_index is not None else 7
        bound = np.ones(int(dim), dtype=np.float64) * joint_delta_bound
        for attr, val in (
            ("input_max", bound),
            ("input_min", -bound),
            ("output_max", bound),
            ("output_min", -bound),
        ):
            if hasattr(ctrl, attr):
                setattr(ctrl, attr, val.copy())
        if joint_kp is not None and hasattr(ctrl, "kp"):
            ctrl.kp = np.ones(int(dim), dtype=np.float64) * joint_kp
        # robosuite caches the scale factor lazily; clear it so it recomputes.
        if hasattr(ctrl, "action_scale"):
            ctrl.action_scale = None
    if not overridden:
        print(
            "[warn] Found robots but no controller object to override; "
            "verify the JOINT_POSITION action scaling on your LIBERO/robosuite version."
        )


def make_joint_position_env(
    bddl_file: Any,
    camera_height: int,
    camera_width: int,
    controller: str = "JOINT_POSITION",
    joint_delta_bound: float = 1.0,
    joint_kp: float | None = None,
) -> Any:
    """Create a LIBERO env configured for joint-position control.

    Tries the ``controller_configs`` kwarg first (robosuite-direct / some LIBERO
    forks), then falls back to the official LIBERO ``controller`` name API and
    overrides the controller scaling in place.
    """
    from libero.libero.envs import OffScreenRenderEnv

    controller_configs = build_joint_position_controller_config(controller, joint_delta_bound, joint_kp)
    if controller_configs is not None:
        try:
            return OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=camera_height,
                camera_widths=camera_width,
                controller_configs=controller_configs,
            )
        except TypeError:
            pass  # LIBERO ControlEnv only accepts a controller name; fall through.

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=camera_height,
        camera_widths=camera_width,
        controller=controller,
    )
    if controller == "JOINT_POSITION":
        _override_controller_scaling(env, joint_delta_bound, joint_kp)
    return env


class PickleWebsocketClient:
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=600,
        )
        self._metadata = pickle.loads(self._ws.recv())

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        payload = dict(obs)
        payload["endpoint"] = "infer"
        self._ws.send(pickle.dumps(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return pickle.loads(response)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        payload = {} if reset_info is None else dict(reset_info)
        payload["endpoint"] = "reset"
        self._ws.send(pickle.dumps(payload))
        self._ws.recv()


class DreamZeroLiberoClient:
    def __init__(
        self,
        host: str,
        port: int,
        open_loop_horizon: int = 8,
        debug_open_loop: bool = False,
        return_video_pred: bool = False,
        reset_server_each_request: bool = False,
        joint_control_mode: str = "delta",
        gripper_threshold: float = 0.02,
        gripper_invert: bool = False,
    ) -> None:
        self.client = PickleWebsocketClient(host=host, port=port)
        self.open_loop_horizon = open_loop_horizon
        self.debug_open_loop = debug_open_loop
        self.return_video_pred = return_video_pred
        self.reset_server_each_request = reset_server_each_request
        self.joint_control_mode = joint_control_mode
        self.gripper_threshold = gripper_threshold
        self.gripper_invert = gripper_invert
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.pred_video_chunks: list[np.ndarray] = []
        self.session_id = str(uuid.uuid4())
        self.request_index = 0
        self.env_step_index = 0
        self._is_first_request = True

    def reset(self) -> None:
        if self.debug_open_loop:
            tqdm.write(f"[client][reset] session={self.session_id} env_steps={self.env_step_index}")
        self.client.reset({"session_id": self.session_id})
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.pred_video_chunks = []
        self.session_id = str(uuid.uuid4())
        self.request_index = 0
        self.env_step_index = 0
        self._is_first_request = True

    def infer(self, obs: dict[str, Any], instruction: str, env_action_dim: int) -> np.ndarray:
        needs_new_chunk = (
            self.actions_from_chunk_completed == 0
            or self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= min(self.open_loop_horizon, len(self.pred_action_chunk))
        )
        if needs_new_chunk:
            if self.reset_server_each_request and not self._is_first_request:
                self.client.reset({"session_id": self.session_id, "reason": "new_chunk_request"})
            self.actions_from_chunk_completed = 0
            self.request_index += 1
            joint, gripper = state(obs)
            result = self.client.infer(
                {
                    "observation/agentview": rgb(first(obs, AGENTVIEW_KEYS)),
                    "observation/eye_in_hand": rgb(first(obs, EYE_IN_HAND_KEYS)),
                    "observation/joint_position": joint,
                    "observation/gripper_position": gripper,
                    "prompt": instruction,
                    "session_id": self.session_id,
                    "client_request_index": self.request_index,
                    "client_env_step_index": self.env_step_index,
                    "client_open_loop_horizon": self.open_loop_horizon,
                    "return_video_pred": self.return_video_pred,
                }
            )
            self._is_first_request = False
            actions = np.asarray(result["actions"], dtype=np.float64)
            if actions.ndim != 2:
                raise ValueError(f"Expected action chunk [N,D], got {actions.shape}")
            self.pred_action_chunk = actions
            if "video_pred" in result:
                self.pred_video_chunks.append(np.asarray(result["video_pred"], dtype=np.uint8))

        target = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed], dtype=np.float64).reshape(-1)
        self.actions_from_chunk_completed += 1
        self.env_step_index += 1

        # The policy returns *absolute* joint targets (relative deltas are already
        # un-relativized server-side). robosuite's JOINT_POSITION controller is
        # delta-based, so command (target - current measured joint) each step.
        current_joint, _ = state(obs)
        current_joint = current_joint.reshape(-1)
        n_joint = current_joint.shape[0]
        joint_target = target[:n_joint]
        if self.joint_control_mode == "delta":
            joint_cmd = joint_target - current_joint
        else:
            joint_cmd = joint_target

        parts = [joint_cmd]
        if target.shape[0] > n_joint:
            gripper_cmd = make_gripper_command(target[n_joint], self.gripper_threshold, self.gripper_invert)
            parts.append(np.asarray([gripper_cmd], dtype=np.float64))
        action = np.concatenate(parts)
        if len(action) < env_action_dim:
            action = np.pad(action, (0, env_action_dim - len(action)))
        return action[:env_action_dim]


def ensure_libero_imports(libero_root: Path) -> None:
    libero_root = libero_root.resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def load_init_states(init_states_path: Path):
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        return torch.load(init_states_path)


def env_action_dim(env: Any) -> int:
    if hasattr(env, "action_spec"):
        return int(env.action_spec[0].shape[0])
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    inner_env = getattr(env, "env", None)
    if inner_env is not None:
        if hasattr(inner_env, "action_spec"):
            return int(inner_env.action_spec[0].shape[0])
        if hasattr(inner_env, "action_dim"):
            return int(inner_env.action_dim)
    raise AttributeError("Could not determine LIBERO environment action dimension")


def make_rollout_frame(obs: dict[str, Any]) -> np.ndarray:
    agentview = np.flipud(rgb(first(obs, AGENTVIEW_KEYS))).copy()
    wrist = np.flipud(rgb(first(obs, EYE_IN_HAND_KEYS))).copy()
    if wrist.shape[0] != agentview.shape[0]:
        row_idx = np.linspace(0, wrist.shape[0] - 1, agentview.shape[0]).astype(np.int64)
        target_width = max(1, int(round(wrist.shape[1] * agentview.shape[0] / wrist.shape[0])))
        col_idx = np.linspace(0, wrist.shape[1] - 1, target_width).astype(np.int64)
        wrist = wrist[row_idx][:, col_idx]
    return np.concatenate([agentview, wrist], axis=1)


def write_rollout_video(frames: list[np.ndarray], output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def write_video_clip(frames: np.ndarray, output_path: Path, fps: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps, codec="libx264")


def write_results(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["task_id", "task_name", "language", "success_rate"])
        writer.writeheader()
        for task in results["tasks"]:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "language": task["language"],
                    "success_rate": task["success_rate"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--benchmark-name", type=str, default="libero_spatial")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--camera-height", type=int, default=160)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/libero_eval"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-video-pred", action="store_true")
    parser.add_argument("--video-episodes-per-task", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--debug-open-loop", action="store_true")
    parser.add_argument("--reset-server-each-request", action="store_true")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument(
        "--joint-control-mode",
        choices=("delta", "absolute"),
        default="delta",
        help="'delta' (default) commands target-minus-current for robosuite's delta JOINT_POSITION controller; "
        "'absolute' sends raw joint targets (only for controllers that accept absolute joint positions).",
    )
    parser.add_argument(
        "--joint-delta-bound",
        type=float,
        default=1.0,
        help="Identity-scaling bound (radians) for the JOINT_POSITION controller; also clips per-step joint deltas.",
    )
    parser.add_argument("--joint-kp", type=float, default=None, help="Optional JOINT_POSITION controller kp override.")
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.02,
        help="Predicted gripper qpos below this is treated as 'closed' (command +1).",
    )
    parser.add_argument("--gripper-invert", action="store_true", help="Flip the gripper open/close sense.")
    parser.add_argument(
        "--num-settle-steps",
        type=int,
        default=10,
        help="Number of zero-action steps after set_init_state to let the scene settle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_libero_imports(args.libero_root)

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark_name)(args.task_order_index)
    task_ids = args.task_ids or list(range(benchmark.n_tasks))
    client = DreamZeroLiberoClient(
        args.host,
        args.port,
        open_loop_horizon=args.open_loop_horizon,
        debug_open_loop=args.debug_open_loop,
        return_video_pred=args.save_video_pred,
        reset_server_each_request=args.reset_server_each_request,
        joint_control_mode=args.joint_control_mode,
        gripper_threshold=args.gripper_threshold,
        gripper_invert=args.gripper_invert,
    )

    results: dict[str, Any] = {
        "benchmark_name": args.benchmark_name,
        "task_order_index": args.task_order_index,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "open_loop_horizon": args.open_loop_horizon,
        "checkpoint_path": str(args.checkpoint_path.resolve()) if args.checkpoint_path is not None else None,
        "server_metadata": client.client.metadata,
        "tasks": [],
    }

    task_progress = tqdm(task_ids, desc="Tasks", unit="task")
    for task_id in task_progress:
        task = benchmark.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = make_joint_position_env(
            bddl_file,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            controller=args.controller,
            joint_delta_bound=args.joint_delta_bound,
            joint_kp=args.joint_kp,
        )
        action_dim = env_action_dim(env)
        init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = load_init_states(init_states_path)

        successes = 0
        task_result: dict[str, Any] = {
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "success_rate": 0.0,
            "episodes": [],
        }
        results["tasks"].append(task_result)

        for episode_idx in tqdm(range(args.n_eval), desc=f"Task {task_id} Episodes", unit="ep", leave=False):
            client.reset()
            env.reset()
            init_state = init_states[episode_idx % len(init_states)]
            if torch.is_tensor(init_state):
                init_state = init_state.cpu().numpy()
            obs = env.set_init_state(init_state)

            # Let the scene settle (objects drop, robot stabilizes) before rollout.
            # A zero action holds the joints (zero delta) and leaves the gripper as-is.
            if args.num_settle_steps > 0:
                settle_action = np.zeros(action_dim, dtype=np.float64)
                for _ in range(args.num_settle_steps):
                    obs, _, _, _ = env.step(settle_action)

            success = False
            steps = 0
            video_frames: list[np.ndarray] = []
            save_video = args.save_video and episode_idx < args.video_episodes_per_task
            save_video_pred = args.save_video_pred and episode_idx < args.video_episodes_per_task
            if save_video:
                video_frames.append(make_rollout_frame(obs))

            while steps < args.max_steps:
                action = client.infer(obs, task.language, action_dim)
                obs, reward, done, info = env.step(action)
                steps += 1
                if save_video:
                    video_frames.append(make_rollout_frame(obs))
                success = bool(info.get("success", False) or reward > 0.5 or env.check_success())
                if success or done:
                    break

            successes += int(success)
            video_path = None
            pred_video_paths: list[str] = []
            if save_video and video_frames:
                video_path = args.output_dir / "videos" / f"task_{task_id:02d}_{task.name}" / f"episode_{episode_idx:03d}.mp4"
                write_rollout_video(video_frames, video_path, fps=args.video_fps)
            if save_video_pred and client.pred_video_chunks:
                pred_dir = args.output_dir / "video_pred" / f"task_{task_id:02d}_{task.name}" / f"episode_{episode_idx:03d}"
                for chunk_idx, chunk in enumerate(client.pred_video_chunks):
                    pred_path = pred_dir / f"request_{chunk_idx:03d}.mp4"
                    write_video_clip(chunk, pred_path, fps=args.video_fps)
                    pred_video_paths.append(str(pred_path))

            task_result["episodes"].append(
                {
                    "episode_index": episode_idx,
                    "success": success,
                    "steps": steps,
                    "video_path": str(video_path) if video_path is not None else None,
                    "video_pred_paths": pred_video_paths,
                }
            )
            task_result["success_rate"] = successes / float(episode_idx + 1)
            results["mean_success_rate"] = float(np.mean([task["success_rate"] for task in results["tasks"]]))
            write_results(args.output_dir, results)

        env.close()
        task_result["success_rate"] = successes / float(args.n_eval)
        results["mean_success_rate"] = float(np.mean([task["success_rate"] for task in results["tasks"]]))
        write_results(args.output_dir, results)

    task_progress.close()
    tqdm.write(f"Mean success rate: {results.get('mean_success_rate', 0.0):.4f}")


if __name__ == "__main__":
    main()
