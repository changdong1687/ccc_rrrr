#!/bin/bash
# DreamZero LIBERO LoRA fine-tuning from a pretrained Wan2.2-5B checkpoint.

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-8}
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"$DREAMZERO_ROOT/data/libero_lerobot_gear"}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$DREAMZERO_ROOT/checkpoints/dreamzero_wan22_5b_pretrained"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_libero_wan22_finetune"}
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
MAX_STEPS=${MAX_STEPS:-50000}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT"
    exit 1
fi
if [ ! -d "$PRETRAINED_CKPT" ]; then
    echo "ERROR: PRETRAINED_CKPT not found at $PRETRAINED_CKPT"
    echo "Set PRETRAINED_CKPT=/path/to/dreamzero_5b_checkpoint"
    exit 1
fi

cd "$DREAMZERO_ROOT"
python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/libero_relative \
    wandb_project=dreamzero-libero \
    train_architecture=lora \
    pretrained_model_path="$PRETRAINED_CKPT" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=5e-6 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.03 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=160 \
    save_lora_only=true \
    max_chunk_size=4 \
    save_strategy=steps \
    libero_data_root="$LIBERO_DATA_ROOT" \
    dit_version="$WAN22_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"
