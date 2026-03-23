TI_BASE_PATH="./diffusers/examples/textual_inversion/training-runs/"
DREAMBOOTH_BASE_PATH="./diffusers/examples/dreambooth/training-runs/"

# Inference for Textual Inversion
( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${TI_BASE_PATH}" --checkpoints_idxs 200 400 --exp_names 00001-40e1-dog6 00002-41t4-dog6 )

# Inference for Textual Inversion with IS/TS segmented images
( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${TI_BASE_PATH}" --checkpoints_idxs 200 400 --with_segmentation --exp_names 00001-40e1-dog6 00002-41t4-dog6 )

# Inference for Dreambooth
( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${DREAMBOOTH_BASE_PATH}" --checkpoints_idxs 200 600 --exp_names 00001-6666-me )
