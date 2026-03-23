#!/bin/bash

inference_type=${1}
config_path=${2}
t=${3}
moft_layers_concept_path=${4}
moft_layers_style_path=${5}
batch_size_medium=${6}

CUDA_VISIBLE_DEVICES=0 python inference_sdxl.py \
  --inference_type="${inference_type}" \
  --config_path="${config_path}" \
  --t="${t}" \
  --guidance_scale=5.0 \
  --version=0 \
  --seed=0 \
  --replace_inference_output \
  --moft_layers_concept_path="${moft_layers_concept_path}" \
  --moft_layers_style_path="${moft_layers_style_path}" \
  --batch_size_medium="${batch_size_medium}" \
