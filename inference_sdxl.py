import argparse
import yaml

from nb_utils.eval_sets import base_set, live_set, object_set, merge_test_set, merge_base_set
from nb_utils.configs import live_object_data
from moft.inferencer_sdxl import inferencers

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to hparams.yml"
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        required=False,
        help="t value for merging"
    )
    parser.add_argument(
        "--parameter",
        type=float,
        default=None,
        required=False,
        help="parameter value for postprocessing"
    )
    parser.add_argument(
        "--postprocessing_method",
        type=str,
        default="curve_over_id",
        required=False,
        help="method for postprocessing the merged matrices"
    )
    parser.add_argument(
        "--num_images_per_medium_prompt",
        type=int,
        default=1,
        help="Number of generated images for each medium prompt",
    )
    parser.add_argument(
        "--num_images_per_base_prompt",
        type=int,
        default=10,
        help="Number of generated images for each base prompt",
    )
    parser.add_argument(
        "--batch_size_medium",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size_base",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--replace_inference_output",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--moft_layers_concept_path",
        type=str,
        default=None,
        required=False,
        help="Path to moft layers concept"
    )
    parser.add_argument(
        "--moft_layers_style_path",
        type=str,
        default=None,
        required=False,
        help="Path to moft layers style"
    )
    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if live_object_data[config['class_name']] == 'live':
        evaluation_set = live_set
    else:
        evaluation_set = object_set
    if args.t is not None or args.inference_type == 'moft_direct_merge':
        evaluation_set = merge_test_set
    print(evaluation_set)
    inferencer = inferencers[args.inference_type](config, args, evaluation_set, merge_base_set)

    inferencer.setup()
    inferencer.generate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
