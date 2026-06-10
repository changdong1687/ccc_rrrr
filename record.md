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
