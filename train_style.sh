#!/bin/bash

trainer_type=${1}
concept=${2}
superclass=${3}
placeholder_token=${4}
nblocks=${5}
moft_scale=${6}

if [ "${moft_scale}" != "False" ]; then
  params+=(--moft_scale)
fi

accelerate launch train.py \
  --test_data_dir="../OrthoFuse/style/${concept}" \
  --train_data_dir="../OrthoFuse/style/${concept}" \
  --class_name="${superclass}" \
  --output_dir="../OrthoFuse/output/concept_style/${trainer_type}" \
  --mixed_precision="no" \
  --trainer_type="${trainer_type}" \
  --train_batch_size=1 \
  --num_train_epochs=1000 \
  --checkpointing_steps=200 \
  --resolution=1024 \
  --num_val_imgs=1 \
  --placeholder_token="${placeholder_token}" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --moft_nblocks="${nblocks}" \
  "${params[@]}"
