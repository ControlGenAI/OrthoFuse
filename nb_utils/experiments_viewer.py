import os
import traceback
from functools import partial
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Tuple, Callable, Optional, Dict, List

import glob
import tqdm
import tqdm.autonotebook

import regex
from natsort import natsorted

import numpy as np

import ipywidgets

import matplotlib as mpl
from seaborn import color_palette


from .utils import _read_config
from .clip_eval import ExpEvaluator
from .images_viewer import create_buttons_grid, _display_widget, MultifolderViewer


class ExpsViewer:
    palette = color_palette('tab10')

    button_colors = {
        (False, False, False): mpl.colors.to_hex(palette[0]),
        (False, False, True): mpl.colors.to_hex(palette[1]),
        (False, True, False): mpl.colors.to_hex(palette[2]),
        (False, True, True): mpl.colors.to_hex(palette[3]),
        (True, False, False): mpl.colors.to_hex(palette[4]),
        (True, False, True): mpl.colors.to_hex(palette[5]),
        (True, True, False): mpl.colors.to_hex(palette[6]),
        (True, True, True): mpl.colors.to_hex(palette[7]),
    }

    def __init__(
            self, base_path: str, ncolumns: int = 5,
            all_info: Dict[Tuple[str, Tuple[str, str, str]], Any] = None, set_name: str = 'medium',
            exp_filter_fn: Callable[[str], bool] = None, lazy_load: bool = True,
            evaluator: Optional[ExpEvaluator] = None, filter_fn: Callable[[str], bool] = None, path_mapping: Optional[dict] = None
    ):
        """
        :param base_path: path to folder with experiments
        :param int ncolumns: number of columns in selector over all experiment
        :param all_info: experiments statistics to draw correct captions for images and figures
        :param set_name: target prompts set for displayed metrics
        :param exp_filter_fn: Filter experiments w.r.t. their names
        :param bool lazy_load: Whether to load all images in class constructor
        :param Optional[ExpEvaluator] evaluator: class to compute IS/TS for a given experiment
        :param Callable[[str], bool] filter_fn: Filter prompts for each experiment
            Also must have a .normalize method (Callable[[str], str]) that will be used to determine order of directories
        """
        self.base_path = base_path
        self.all_info = all_info
        self.set_name = set_name

        self.lazy_load = lazy_load
        self.evaluator = evaluator
        self.filter_fn = filter_fn

        self.exps_names = sorted([
            name for name in os.listdir(base_path)
            if regex.match('[0-9]+-.*-.*', name)
        ])
        if exp_filter_fn is not None:
            self.exps_names = list(filter(exp_filter_fn, self.exps_names))

        self._configs = {name: self._read_config(name, path_mapping) for name in self.exps_names}

        exps_grid_size = np.array([(len(self.exps_names) - 1) // ncolumns + 1, ncolumns])
        buttons_grid, buttons = create_buttons_grid(exps_grid_size)

        dreambooth_legend = [ipywidgets.Button() for _ in ExpsViewer.button_colors]
        for button, (spec, button_color) in zip(dreambooth_legend, ExpsViewer.button_colors.items()):
            button.description = '\n'.join(
                ['+TE'] * spec[0] +
                ['+PL'] * spec[1] +
                ['+CN'] * spec[2]
            )
            button.style.button_color = button_color

        self.exps_views = {}
        output = ipywidgets.Output()
        for name, button in zip(self.exps_names, buttons):
            button.description = name
            button.on_click(partial(self._load_exp_view, name=name, output=output))

            # Define button color
            button_color = None

            config = self._configs[name]

            samples_dirs = natsorted(
                glob.glob(os.path.join(config['output_dir'], 'checkpoint-*', 'samples', '*', '*'))
            )
            if len(samples_dirs) == 0:
                button_color = 'black'

            is_dreambooth, is_textual_inversion, is_custom_diff = 'DB' in config['exp_name'], 'TI' in config['exp_name'], 'CD' in config['exp_name']
            if is_dreambooth:
                train_text_encoder = config['train_text_encoder']
                with_prior_preservation = config['with_prior_preservation']
                with_class_name = config['instance_prompt'] != 'a photo of sks'

                button_color = ExpsViewer.button_colors[(train_text_encoder, with_prior_preservation, with_class_name)]
            if is_textual_inversion:
                with_class_name = False
                button_color = 'green'
            if is_custom_diff:
                button_color = 'purple'
            if config.get('attention_masks_dir') is not None:
                button.style.text_color = 'green'
            if config.get('prompts_attention_masks_dir') is not None:
                button.style.text_color = 'yellow'
                button.tooltip = f"{name} ({config['prompts_attention_map_mode']})"
            if (
                    config.get('attention_masks_dir') is not None and
                    config.get('prompts_attention_masks_dir') is not None
            ):
                button.style.text_color = 'red'
                button.tooltip = f"{name} ({config['prompts_attention_map_mode']})"

            button.style.button_color = button_color

        self.widget = ipywidgets.VBox([
            ipywidgets.HBox(dreambooth_legend),
            buttons_grid,
            output
        ])

    @staticmethod
    def _view_exp(exp_name, config, all_info, set_name, lazy_load=True, filter_fn=None):
        output = ipywidgets.Output()
        samples_dirs = natsorted(glob.glob(os.path.join(config['output_dir'], 'checkpoint-*', 'samples', '*', 'version_*')))
        samples_dirs_buttons_grid, samples_dirs_buttons = create_buttons_grid([(len(samples_dirs) - 1) // 2 + 1, 2])

        for samples_dir, button in (pbar := tqdm.autonotebook.tqdm(
                zip(samples_dirs, samples_dirs_buttons), total=len(samples_dirs), leave=False, disable=True
        )):
            [*_, checkpoint_idx, _, sampling_config, _] = os.path.normpath(samples_dir).split(os.path.sep)
            pbar.set_description(checkpoint_idx)

            [checkpoint_idx] = regex.findall('checkpoint-([0-9]+)', checkpoint_idx)
            (num_inference_steps, guidance_scale, other_inference_specs) = regex.match(
                'ns([0-9]+)_gs([^_]+)(?:_(.*))?', sampling_config
            ).groups()
            other_inference_specs = [] if other_inference_specs is None else other_inference_specs.split('_')

            button.description = 'ckpt: {0}, st: {1}, gs: {2}, others: {3}'.format(
                checkpoint_idx, num_inference_steps, guidance_scale, other_inference_specs
            )
            info = None
            specs = (checkpoint_idx, num_inference_steps, guidance_scale, *other_inference_specs)
            if all_info is not None and (exp_name, specs) in all_info:
                info = all_info[(exp_name, specs)]

            exp_widget = MultifolderViewer(samples_dir, lazy_load=lazy_load, info=info, filter_fn=filter_fn, set_name=set_name).view(ncolumns=5)
            button.on_click(partial(_display_widget, output=output, widget=exp_widget))

        return ipywidgets.VBox([
            ipywidgets.Label(value=config['output_dir']),
            samples_dirs_buttons_grid,
            output
        ])

    def _load_exp_view(self, button, name, output):
        with output:
            if name not in self.exps_views:
                config = self._configs[name]
                self.exps_views[name] = self._view_exp(name, config, self.all_info, self.set_name, self.lazy_load, self.filter_fn)
            _display_widget(button, output, self.exps_views[name])

    def view(self):
        return self.widget

    def _read_config(self, exp_name, path_mapping):
        return _read_config(self.base_path, exp_name, path_mapping)

    def get_samples_dirs(self):
        """ For all experiments determine all folders with samples (for all checkpoint idxs, num_inference_steps and guidance_scales)
        :return: dictionary where for each experiment all its folders with samples are stored
        """
        exp_samples_dirs: defaultdict[str, defaultdict[Tuple[str, ...], Any]]
        exp_samples_dirs = defaultdict(lambda: defaultdict(str))
        for name in self.exps_names:
            config = self._configs[name]

            samples_dirs = natsorted(glob.glob(
                os.path.join(config['output_dir'], 'checkpoint-*', 'samples', '*', '*')
            ))
            for samples_dir in samples_dirs:
                [*_, checkpoint_idx, _, sampling_config, _] = os.path.normpath(samples_dir).split(os.path.sep)

                [checkpoint_idx] = regex.findall('checkpoint-([0-9]+)', checkpoint_idx)
                (num_inference_steps, guidance_scale, other_inference_specs) = regex.match(
                    'ns([0-9]+)_gs([^_]+)(?:_(.*))?', sampling_config
                ).groups()
                other_inference_specs = [] if other_inference_specs is None else other_inference_specs.split('_')

                exp_samples_dirs[name][(checkpoint_idx, num_inference_steps, guidance_scale, *other_inference_specs)] = (
                    samples_dir, config
                )

        return exp_samples_dirs

    def _evaluate(
            self, exp_name: str, checkpoint_idx: str, inference_specs: Tuple[str, ...] = ('50', '7.5'),
            cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None, verbose: bool = False
    ) -> Optional[Tuple[str, Dict]]:
        """ Evaluate CLIP and DINO IS/TS for a given experiment
        :param str exp_name: target experiment
        :param str checkpoint_idx: target checkpoint idx
        :param Tuple[str, ...] inference_specs: target num inference steps, guidance scale, and other inference specs
        :param Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] cache: dictionary with cached statistics.
            All keys are of form (exp_name, (checkpoint_idx, num_inference_steps, guidance_scale))
        :param bool verbose: whether to log errors or not
        :return: experiment name and all its statistics
        """
        specs: Tuple[str, ...] = (checkpoint_idx, *inference_specs)
        try:
            samples_dirs = self.get_samples_dirs()
            samples_path, config = samples_dirs[exp_name][specs]

            # if cache is not None and (exp_name, specs) in cache:
            #     n_evaluated_prompts = len(cache[(exp_name, specs)].get('dino_image_similarities', []))

            #     if n_evaluated_prompts != len(os.listdir(samples_path)):
            #         print(f'Reevaluate {exp_name, specs} for new samples')
            #     else:
            #         print(f'Use cache for {exp_name, specs}')
            #         return exp_name, cache[(exp_name, specs)]
            exp_viewer = MultifolderViewer(samples_path, lazy_load=False)

            results = self.evaluator(exp_viewer, config, verbose=verbose)

            return exp_name, {**results, **{'specs': specs, 'config': config}}
        except Exception as ex:
            print(f'Failure evaluating {exp_name}, {checkpoint_idx}:', ex, traceback.format_exc())
            return f'', {'specs': specs}

    def evaluate(
            self, exps_names: List[str], checkpoint_idx: str, inference_specs: Tuple[str, ...],
            processes: int = 0, cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None, verbose: bool = False
    ) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
        """ Multithreading implementation of experiments evaluation
        :param List[str] exps_names: list of experiments to evaluate
        :param str checkpoint_idx: target checkpoint idx
        :param Tuple[str, ...] inference_specs: target num inference steps, guidance scale, and other inference specs
        :param int processes: maximum number of parallel processes
        :param Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] cache: dictionary with cached statistics.
            All keys are of form (exp_name, (checkpoint_idx, *inference_specs))
                for example: (exp_name, (checkpoint_idx, num_inference_steps, guidance_scale))
        :param bool verbose: whether to log errors or not
        :return: statistics for experiments in the same form as cache
        """
        if isinstance(exps_names, str):
            exps_names = [exps_names]

        all_stats = {}
        eval_fn = partial(
            self._evaluate, checkpoint_idx=checkpoint_idx, inference_specs=inference_specs,
            cache=cache, verbose=verbose
        )

        if processes > 0:
            pool = ThreadPool(processes=processes)
            mapper = pool.imap
        else:
            mapper = map

        for exp_name, results in tqdm.tqdm(mapper(eval_fn, exps_names), total=len(exps_names)):
            if exp_name == '':
                continue
            all_stats[(exp_name, tuple(results['specs']))] = results

        if processes > 0:
            # noinspection PyUnboundLocalVariable
            pool.close()
            pool.join()

        return all_stats
