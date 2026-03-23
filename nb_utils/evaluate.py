import os
from argparse import ArgumentParser

import torch

from .clip_eval import ExpEvaluator
from .experiments_viewer import ExpsViewer
from .cache import Cache, DistributedCache


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help='GPU device'
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=6,
        required=False,
        help='Number of parallel threads to perform evaluation'
    )
    parser.add_argument(
        '--exp_names',
        type=str,
        nargs='+',
        required=True,
        help='Target experiments'
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help='Path to the folder with experiments'
    )
    parser.add_argument(
        "--with_segmentation",
        action="store_true",
        default=False,
        help='Whether to calculate IS/TS with masking'
    )
    parser.add_argument(
        "--checkpoints_idxs",
        type=int,
        nargs='+',
        required=True,
        help='Target checkpoint idxs'
    )
    parser.add_argument(
        "--cache_files_template",
        type=str,
        default='./diffusers/examples/*/training-runs/*/evaluate.cache',
        required=False,
        help='Template for existing cache files'
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tau",
        type=str,
        default=None,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda', args.gpu)
    print(f'Evaluating for {args.base_path}')
    print(f'Run evaluation for names: {args.exp_names} for checkpoints: {args.checkpoints_idxs}')

    evaluator = ExpEvaluator(device)
    all_cache = DistributedCache(args.cache_files_template).get()

    exps_viewer = ExpsViewer(
        base_path=args.base_path,
        exp_filter_fn=lambda x: x in args.exp_names,
        ncolumns=6, lazy_load=True, evaluator=evaluator
    )
    for checkpoint_idx in args.checkpoints_idxs:
        if args.beta is not None:
            inference_specs = ('50', '6.0', args.beta, args.tau)
        else:
            inference_specs = ('50', '6.0')
        stats = exps_viewer.evaluate(
            exps_names=args.exp_names, checkpoint_idx=str(checkpoint_idx), inference_specs=('50', '6.0', args.beta, args.tau),
            cache=all_cache, processes=args.processes, with_segmentation=args.with_segmentation
        )
        for key, value in stats.items():
            if 'config' not in value:
                continue
            exp_cache = Cache(os.path.join(value['config']['output_dir'], 'evaluate.cache'))
            exp_cache.update({key: value})
