# LIBERO 过拟合 sanity check

目的：**只用一条任务（甚至一条 episode）的数据训练，再用同一份数据评测**，验证在过拟合条件下模型能否学到正确的关节动作。

核心工具是**离线 teacher-forcing 对比**（`eval_overfit_offline.py`）：不跑仿真，把训练数据里的真实观测喂进模型，把预测动作和数据集里的真值动作（`next_state` 绝对关节角）直接比 MAE。这样能把两件事解耦：

- **预测 ≈ 真值** → 模型确实学到了动作映射；之后 sim 里抓不到，问题在控制器 / 夹爪 / 仿真执行。
- **预测 ≠ 真值（即便过拟合）** → 问题在训练 / 数据本身。

> 本目录不修改任何现有代码：训练脚本只是用环境变量包一层调用现有的 `../train_libero_wan22.sh`；离线 eval 直接 import 现有的 `../eval_libero_server.py` 里的 `LiberoJointPolicy` 和 `GrootSimPolicy`。

---

## 步骤 1：构造极小数据集（二选一）

**(a) 一整条任务（~50 条 demo）——用现有转换脚本**，`--input` 指向单个任务的 hdf5：

```bash
python simulation/libero/convert_libero_data.py \
  --input /path/to/LIBERO/.../<one_task>_demo.hdf5 \
  --output ./data/libero_debug_onetask \
  --force
```

**(b) 单条 episode（过拟合最彻底）——从已转换的数据集里抽子集**：

```bash
python simulation/libero/debug/make_debug_dataset.py \
  --src ./data/libero_spatial \
  --dst ./data/libero_debug_onetask \
  --num-episodes 1 --force
# 或指定某几条： --episode-indices 0 1 2
```

## 步骤 2：在极小数据集上训练

```bash
LIBERO_DATA_ROOT=./data/libero_debug_onetask \
OUTPUT_DIR=./checkpoints/dreamzero_libero_overfit_debug \
MAX_STEPS=5000 NUM_GPUS=1 \
bash simulation/libero/debug/train_overfit.sh
```

过拟合一般几千步就能看出趋势；loss 应明显下降到很低。

## 步骤 3：离线对比预测动作 vs 真值动作（关键）

```bash
python simulation/libero/debug/eval_overfit_offline.py \
  --model-path ./checkpoints/dreamzero_libero_overfit_debug/checkpoint-5000 \
  --dataset-root ./data/libero_debug_onetask \
  --episode-index 0 \
  --output ./runs/overfit_debug
```

输出 `overfit_summary.json` + `overfit_pred_vs_gt.npz`，关注：

- `joint_mae_overall`（弧度）：很小（经验上 < ~0.02）→ 模型复现了训练动作 → **学习链路 OK**；偏大 → 训练 / 数据有问题。
- `chunk_mse_mean`：整段预测 chunk 对齐真值的 MSE。
- `gripper_pred_min/max` vs `gripper_gt_min/max`：用来**校准 sim eval 的 `--gripper-threshold`**（看预测夹爪值的真实分布落在哪）。

## 步骤 4（可选）：在同一条数据上跑真·仿真

离线对比通过后，再用现有 client/server 或 `eval_libero.py` 跑这条任务的 sim rollout：

- 若离线预测准、但 sim 抓不到 → 问题在控制器 / 夹爪映射（调 `JOINT_KP`、`--gripper-threshold` / `--gripper-invert`、`OPEN_LOOP_HORIZON`）。
- 若离线预测也不准 → 回到训练 / 数据。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `eval_overfit_offline.py` | 离线 teacher-forcing：预测动作 vs 真值动作对比（不跑仿真） |
| `train_overfit.sh` | 用环境变量包一层，调用现有 `train_libero_wan22.sh` 在极小数据集上训练 |
| `make_debug_dataset.py` | 从已转换数据集抽取 N 条 episode 成子集（含 meta 重写） |
