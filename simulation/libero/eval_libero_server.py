#!/usr/bin/env python3
"""Serve DreamZero as a websocket policy for LIBERO joint-action evaluation."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import logging
import os
import pickle
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import websockets
import websockets.asyncio.server
import websockets.frames
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


@dataclasses.dataclass
class ServerMetadata:
    embodiment: str = "libero_sim"
    action_space: str = "joint_position"
    expected_views: int = 2


INFER_SIGNAL = 0
SHUTDOWN_SIGNAL = 1
RESET_SIGNAL = 2


class LiberoJointPolicy:
    """Small websocket adapter around GrootSimPolicy for LIBERO joint observations."""

    FRAMES_PER_CHUNK = 4

    def __init__(self, policy: GrootSimPolicy) -> None:
        self._policy = policy
        self._debug_open_loop = os.environ.get("DREAMZERO_DEBUG_OPEN_LOOP", "0") == "1"
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.agentview": [],
            "video.eye_in_hand": [],
        }
        self._is_first_call = True
        self._current_session_id: str | None = None

    def _reset_action_head_state(self) -> None:
        action_head = self._policy.trained_model.action_head
        for name in (
            "current_start_frame",
            "language",
            "clip_feas",
            "ys",
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
        ):
            if hasattr(action_head, name):
                setattr(action_head, name, 0 if name == "current_start_frame" else None)

    def _reset_local_buffers(self) -> None:
        for frames in self._frame_buffers.values():
            frames.clear()
        self._is_first_call = True

    def reset(self, payload: dict[str, Any]) -> None:
        self._reset_action_head_state()
        self._reset_local_buffers()
        self._current_session_id = None
        if self._debug_open_loop:
            print(f"[server][reset] session={payload.get('session_id', 'unknown')}")

    def _ensure_session(self, obs: dict[str, Any]) -> None:
        session_id = obs.get("session_id")
        if session_id is None or session_id == self._current_session_id:
            return
        if self._current_session_id is not None:
            self._reset_action_head_state()
            self._reset_local_buffers()
        self._current_session_id = session_id

    def _accumulate_video(self, key: str, value: Any) -> np.ndarray:
        video = np.asarray(value, dtype=np.uint8)
        if video.ndim == 4:
            return video
        if video.ndim != 3:
            raise ValueError(f"Expected video input with 3 or 4 dims, got shape {video.shape}")

        buffer = self._frame_buffers[key]
        buffer.append(video)
        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK
        if len(buffer) >= num_frames:
            frames_to_use = buffer[-num_frames:]
        else:
            frames_to_use = buffer.copy()
            while len(frames_to_use) < num_frames:
                frames_to_use.insert(0, buffer[0])
        return np.stack(frames_to_use, axis=0)

    @staticmethod
    def _as_state(value: Any, width: int | None = None) -> np.ndarray:
        state = np.asarray(value, dtype=np.float64)
        if state.ndim == 1:
            state = state[None, ...]
        if state.ndim != 2:
            raise ValueError(f"Expected state input with 1 or 2 dims, got shape {state.shape}")
        if width is not None:
            state = state[:, :width]
        return state

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "video.agentview": self._accumulate_video("video.agentview", obs["observation/agentview"]),
            "video.eye_in_hand": self._accumulate_video("video.eye_in_hand", obs["observation/eye_in_hand"]),
            "state.joint_position": self._as_state(obs["observation/joint_position"]),
            "state.gripper_position": self._as_state(obs["observation/gripper_position"], width=1),
            "annotation.task": obs.get("prompt", ""),
        }

    def _forward(self, obs: dict[str, Any]):
        converted = self._convert_observation(obs)
        with torch.inference_mode():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(Batch(obs=converted))
        return result_batch, video_pred

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _format_actions(self, action_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        joint = action_dict.get("action.joint_position")
        gripper = action_dict.get("action.gripper_position")
        if joint is None:
            raise KeyError(f"Policy output has no action.joint_position. Keys: {list(action_dict.keys())}")

        joint_arr = self._to_numpy(joint).astype(np.float32)
        if joint_arr.ndim == 1:
            joint_arr = joint_arr[None, :]
        elif joint_arr.ndim > 2:
            joint_arr = joint_arr.reshape(-1, joint_arr.shape[-1])

        parts = [joint_arr]
        if gripper is not None:
            gripper_arr = self._to_numpy(gripper).astype(np.float32)
            if gripper_arr.ndim == 0:
                gripper_arr = gripper_arr.reshape(1, 1)
            elif gripper_arr.ndim == 1:
                gripper_arr = gripper_arr[:, None]
            elif gripper_arr.ndim > 2:
                gripper_arr = gripper_arr.reshape(-1, gripper_arr.shape[-1])
            parts.append(gripper_arr[:, :1])
        return {"actions": np.concatenate(parts, axis=-1)}

    def _decode_video_pred(self, video_pred: torch.Tensor) -> np.ndarray:
        action_head = self._policy.trained_model.action_head
        vae = action_head.vae
        with torch.inference_mode():
            decoded = vae.decode(video_pred.to(device=action_head._device, dtype=torch.bfloat16))
        decoded = decoded.detach().float().cpu()
        if decoded.ndim != 5:
            raise ValueError(f"Expected decoded video [B,C,T,H,W], got {decoded.shape}")
        decoded = decoded[0].permute(1, 2, 3, 0).numpy()
        return ((decoded + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ensure_session(obs)
        if self._debug_open_loop:
            print(
                f"[server][infer] session={obs.get('session_id', 'unknown')} "
                f"request={obs.get('client_request_index', 'unknown')} "
                f"env_step={obs.get('client_env_step_index', 'unknown')}"
            )
        result_batch, video_pred = self._forward(obs)
        formatted = self._format_actions(result_batch.act)
        self._is_first_call = False
        if bool(obs.get("return_video_pred", False)) and video_pred is not None:
            formatted["video_pred"] = self._decode_video_pred(video_pred)
        return formatted

    def participate(self, obs: dict[str, Any]) -> None:
        self._ensure_session(obs)
        self._forward(obs)
        self._is_first_call = False


class PicklePolicyServer:
    def __init__(
        self,
        policy: LiberoJointPolicy,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        signal_group: dist.ProcessGroup | None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._rank = rank
        self._world_size = world_size
        self._signal_group = signal_group
        self._broadcast_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        if self._rank == 0:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    def _broadcast_signal(self, signal: int) -> None:
        if self._world_size == 1:
            return
        signal_tensor = torch.tensor([signal], dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

    def _broadcast_payload(self, payload: dict[str, Any]) -> None:
        if self._world_size == 1:
            return
        serialized = pickle.dumps(payload)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device=self._broadcast_device)
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.tensor(list(serialized), dtype=torch.uint8, device=self._broadcast_device)
        dist.broadcast(data_tensor, src=0)

    def _receive_payload(self) -> dict[str, Any]:
        size_tensor = torch.zeros(1, dtype=torch.int64, device=self._broadcast_device)
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device=self._broadcast_device)
        dist.broadcast(data_tensor, src=0)
        return pickle.loads(data_tensor.cpu().numpy().tobytes())

    def _distributed_reset(self, payload: dict[str, Any]) -> None:
        if self._world_size > 1:
            self._broadcast_signal(RESET_SIGNAL)
            self._broadcast_payload(payload)
        self._policy.reset(payload)

    def _distributed_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._world_size == 1:
            return self._policy.infer(payload)
        self._broadcast_signal(INFER_SIGNAL)
        self._broadcast_payload(payload)
        dist.barrier()
        result = self._policy.infer(payload)
        dist.barrier()
        return result

    async def _worker_loop(self) -> None:
        logging.info("Rank %d entering distributed worker loop", self._rank)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        while True:
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            signal = int(signal_tensor.item())
            if signal == SHUTDOWN_SIGNAL:
                break
            payload = self._receive_payload()
            if signal == RESET_SIGNAL:
                self._policy.reset(payload)
                continue
            if signal != INFER_SIGNAL:
                raise RuntimeError(f"Unsupported distributed signal: {signal}")
            dist.barrier()
            self._policy.participate(payload)
            dist.barrier()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        await websocket.send(pickle.dumps(dataclasses.asdict(ServerMetadata())))
        while True:
            try:
                payload = pickle.loads(await websocket.recv())
                endpoint = payload.pop("endpoint")
                if endpoint == "reset":
                    self._distributed_reset(payload)
                    await websocket.send(pickle.dumps({"status": "reset successful"}))
                elif endpoint == "infer":
                    result = self._distributed_infer(payload)
                    await websocket.send(pickle.dumps(result))
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-chunk-size", type=int, default=None)
    parser.add_argument("--num-frame-per-block", type=int, default=None)
    parser.add_argument("--debug-open-loop", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    return parser.parse_args()


def init_runtime(device_arg: str, timeout_seconds: int) -> tuple[str, int, int, DeviceMesh | None, dist.ProcessGroup | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if dist.is_initialized():
        rank = dist.get_rank()
    elif world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        rank = 0

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-GPU mode requires CUDA.")
        torch.cuda.set_device(local_rank)
        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))
        signal_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=timeout_seconds))
        return "cuda", rank, world_size, device_mesh, signal_group
    return device_arg, rank, 1, None, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    os.environ["DREAMZERO_DEBUG_OPEN_LOOP"] = "1" if args.debug_open_loop else "0"
    torch._dynamo.config.recompile_limit = 800
    torch._dynamo.config.cache_size_limit = 1000

    model_config_overrides: list[str] = []
    if args.max_chunk_size is not None:
        model_config_overrides.append(f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={args.max_chunk_size}")
    if args.num_frame_per_block is not None:
        model_config_overrides.append(f"action_head_cfg.config.num_frame_per_block={args.num_frame_per_block}")
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.num_frame_per_block={args.num_frame_per_block}"
        )

    device, rank, world_size, device_mesh, signal_group = init_runtime(args.device, args.timeout_seconds)
    logging.info("Initialized rank %d/%d on device %s", rank, world_size, device)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_SIM,
        model_path=args.model_path,
        device=device,
        model_config_overrides=model_config_overrides,
        device_mesh=device_mesh,
    )
    server = PicklePolicyServer(
        LiberoJointPolicy(policy),
        host=args.host,
        port=args.port,
        rank=rank,
        world_size=world_size,
        signal_group=signal_group,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
