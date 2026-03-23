#!/bin/bash

source deactivate
source activate vsdiffusion

OUTPUT_DIR="/home/aalanov/vsoboleva/vs/res/"

# exp_name=${1}
# checkpoint_idx=${2}
beta=${1}
tau=${2}

( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${OUTPUT_DIR}" --beta "${beta}" --tau "${tau}" --checkpoints_idxs 25 200 --exp_names 00033-e7b3-dog6 )
