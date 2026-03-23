#!/bin/bash

concept_exp_name="dog6"
style_list=("01_08")

for style in "${style_list[@]}"; do
  echo "Processing style: ${style}"
  for t in 0.6; do
    echo "Processing t: ${t}"
    ./orthofuse_inference_merge_sdxl.sh moft_merge_fast \
      "./output/concept_style/sdxl_merge/example/logs/hparams.yml" \
      "${t}" \
      "./output/concept_style/sdxl_merge/example/pytorch_lora_weights_concept.safetensors" \
      "./output/concept_style/sdxl_merge/example/pytorch_lora_weights_style.safetensors" \
      "1" \

  done
done

