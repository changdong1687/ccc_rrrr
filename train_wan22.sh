  export DREAMZERO_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr
  export LIBERO_DATA_ROOT=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/data/libero_spatial

  export WAN22_CKPT_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/checkpoints/Wan2.2-TI2V-5B
  export IMAGE_ENCODER_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/checkpoints/Wan2.1-I2V-14B-480P
  export TOKENIZER_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/checkpoints/umt5-xxl

  export OUTPUT_DIR=/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/ccc_rrrr/checkpoints/dreamzero_libero_wan22_lora_smoke_gbs4

  export NUM_GPUS=1
  export PER_DEVICE_BS=4
  export MAX_STEPS=100
  export WANDB_MODE=offline

  bash simulation/libero/train_libero_wan22.sh