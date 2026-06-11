### Wan2.2-5B
```
Loading relative action stats from /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/data/libero_spatial/meta/relative_stats_dreamzero.json
relative_action_per_horizon False
Using relative stats for joint_position: {'max': array([0.46448535, 1.10493329, 0.59629082, 1.53660098, 0.92250961,
       1.27043545, 1.33598959]), 'min': array([-0.32371881, -0.63693237, -0.45711735, -0.95991561, -1.14854248,
       -0.98645949, -1.11442316]), 'mean': array([ 0.02995756,  0.1373611 ,  0.00702643,  0.14241197, -0.02781391,
        0.00171817,  0.07532857]), 'std': array([0.06266888, 0.21130294, 0.07303756, 0.26685632, 0.12870812,
       0.21469389, 0.18929385]), 'q01': array([-0.12733212, -0.24345539, -0.17514194, -0.40103487, -0.43991076,
       -0.55898214, -0.44443138]), 'q99': array([0.20708085, 0.71861853, 0.24322135, 0.91676752, 0.33008996,
       0.57626678, 0.60211244])}
Initialized dataset libero_spatial with EmbodimentTag.LIBERO_SIM
Generated 7 shards for dataset /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/data/libero_spatial
Time taken to initialize 1 datasets: 0.85 seconds
Initializing datasets: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.17it/s]
Using dataset:
Mixture dataset:
- Dataset: libero_spatial (62250 steps)
  Max shard length: 8981
  Min shard length: 8788
  Num shards: 7
  Sampling weight: 1.0
Rank: 0
World size: 1
```

### Eval
```bash
export PYTHONPATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0


python simulation/libero/eval_libero.py \
  --model_path ./checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-4000 \
  --benchmark libero_spatial \
  --bddl-root /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO/libero/libero/bddl_files/libero_spatial \
  --episodes-per-task 10 \
  --max-steps 500 \
  --device cuda:0 \
  --output-dir ./results_libero_spatial

```


```bash
export PYTHONPATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset EGL_DEVICE_ID

python simulation/libero/eval_libero.py \
  --model_path ./checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-4000 \
  --benchmark libero_spatial \
  --bddl-root /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO/libero/libero/bddl_files/libero_spatial \
  --episodes-per-task 10 \
  --max-steps 500 \
  --device cuda:0 \
  --output-dir ./results_libero_spatial \
  --save-videos

```

---

## LIBERO 评测完整说明（关节动作空间）

> 模型预测的是**绝对关节目标 + 夹爪 qpos**（训练用相对关节动作，推理时已 un-relativize 加回当前状态）。
> 因此 eval 必须用 **JOINT_POSITION 控制器**，并把绝对目标转成 `delta = 目标 − 当前实测` 下发（robosuite 的 JOINT_POSITION 是增量式），夹爪 qpos 二值化成 ±1。
> 这些逻辑已内置在 `eval_libero.py` / `eval_libero_client.py`，相关参数都有默认值，开箱即用。

### 0. 渲染环境变量（必须先设）

`eval.txt` 里的 `Cannot initialize a EGL device display` 是离屏渲染初始化失败，用下面这组环境变量解决（换卡时改 `MUJOCO_EGL_DEVICE_ID`）：

```bash
export PYTHONPATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset EGL_DEVICE_ID
```

### 方式 A：单进程 `eval_libero.py`（推荐先用它验证）

设好上面的环境变量后：

```bash
cd /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr

python simulation/libero/eval_libero.py \
  --model_path ./checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-4000 \
  --benchmark libero_spatial \
  --bddl-root /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO/libero/libero/bddl_files/libero_spatial \
  --episodes-per-task 10 \
  --max-steps 500 \
  --device cuda:0 \
  --output-dir ./results_libero_spatial \
  --save-videos
```

关节控制/夹爪/settle 参数都有默认值（`--controller JOINT_POSITION`、`--joint-control-mode delta`、
`--joint-delta-bound 1.0`、`--gripper-threshold 0.02`、`--num-settle-steps 10`），不加也是修好后的行为。
夹爪开合方向反了就加 `--gripper-invert`。

### 方式 B：server + client（websocket，多卡或复用 server）

**终端 1 — server**（只跑策略，不渲染，无需 EGL 变量）：

```bash
cd /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr
MODEL_PATH=./checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-4000 \
DEVICE=cuda:0 \
bash simulation/libero/eval_libero_server.sh
# 多卡： NUM_GPUS=4 MODEL_PATH=... bash simulation/libero/eval_libero_server.sh
```

等 server 加载完模型、开始监听后，再起 client。

**终端 2 — client**（要渲染，先设第 0 步的 EGL 变量）：

```bash
cd /inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr
export PYTHONPATH=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset EGL_DEVICE_ID

CHECKPOINT_PATH=./checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-4000 \
BENCHMARK_NAME=libero_spatial \
N_EVAL=10 MAX_STEPS=500 \
SAVE_VIDEO=true DEBUG_OPEN_LOOP=true \
bash simulation/libero/eval_libero_client.sh
```

`eval_libero_client.sh` 可用环境变量（均有默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CHECKPOINT_PATH` | 自动找最新 checkpoint-* | 模型路径 |
| `LIBERO_ROOT` | 已指向集群 LIBERO | LIBERO 仓库路径 |
| `BENCHMARK_NAME` | `libero_spatial` | 评测套件 |
| `N_EVAL` / `MAX_STEPS` | 20 / 500 | 每任务回合数 / 最大步数 |
| `OPEN_LOOP_HORIZON` | 8 | 每次推理执行的动作步数 |
| `CONTROLLER` | `JOINT_POSITION` | 控制器类型 |
| `JOINT_CONTROL_MODE` | `delta` | `delta`=目标−当前；`absolute`=绝对关节角 |
| `JOINT_DELTA_BOUND` | 1.0 | 恒等缩放上限 / 每步 delta 截断 (rad) |
| `JOINT_KP` | 空 | 可选覆盖控制器 kp |
| `GRIPPER_THRESHOLD` | 0.02 | 预测夹爪 qpos 低于此判为「关」(+1) |
| `GRIPPER_INVERT` | false | 翻转夹爪开合方向 |
| `NUM_SETTLE_STEPS` | 10 | set_init_state 后稳定步数 |
| `SAVE_VIDEO` / `SAVE_VIDEO_PRED` | false | 存 rollout 视频 / 模型预测视频 |
| `DEBUG_OPEN_LOOP` | false | 打印逐次请求调试信息 |

脚本末尾保留 `"$@"` 透传，可临时追加任意 `--xxx` 参数。

### 调试顺序与注意事项

1. **先跑方式 A + `--save-videos`**，确认：
   - 启动日志里**没有** `[warn] ... override ...`（出现说明本机 robosuite 控制器对象路径不同，恒等缩放没生效，机械臂会动得很慢够不到目标，需按版本调整 `_override_controller_scaling`）；
   - 机械臂正常运动、夹爪开合方向正确（反了加 `--gripper-invert` / `GRIPPER_INVERT=true`，阈值不合适调 `--gripper-threshold`）。
2. 校准 OK 后再用大的 `N_EVAL` 跑正式评测。
3. **分辨率**：默认相机 160×320 对齐 wan22 训练；若用 wan2.1 checkpoint，改成 `--camera-height 176 --camera-width 320`（client 用 `CAMERA_HEIGHT=176`）。
4. 结果输出：方式 A 写 `results.jsonl` + `summary.json`；方式 B 写 `results.json` + `results.csv`（含每任务成功率与 `mean_success_rate`）。
