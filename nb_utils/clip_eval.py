"""Based on https://github.com/rinongal/textual_inversion/blob/main/evaluation/clip_eval.py"""
import os
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

import clip

import numpy as np
import pandas as pd

import torch
import torch.backends.cuda
from torchvision import transforms

import PIL
from PIL.Image import Image

from .eval_sets import evaluation_sets
from .images_viewer import MultifolderViewer


class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-1.0, -1.0, -1.0],
                    std=[2.0, 2.0, 2.0])
            ] +                               # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1]
            clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
            clip_preprocess.transforms[4:]    # + skip convert PIL to tensor
        )
        for transform in self.preprocess.transforms:
            if isinstance(transform, transforms.Resize):
                transform.antialias = False

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        similarity_matrix = src_img_features @ gen_img_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        similarity_matrix = text_features @ gen_img_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()


class DINOEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32', dino_model='dinov2_vits14') -> None:
        super().__init__(device, clip_model=clip_model)

        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model).to(self.device)

        self.dino_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.no_grad()
    def dino_encode_images(self, images: List[Image]) -> torch.Tensor:
        images = torch.stack([self.dino_preprocess(image) for image in images])
        images = images.to(self.device)
        return self.dino_model(images)

    def get_dino_image_features(self, img: List[Image], norm: bool = True) -> torch.Tensor:
        image_features = self.dino_encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


class ExpEvaluator:
    def __init__(self, device):
        self.device = device
        self.evaluator = DINOEvaluator(device=device)

    @staticmethod
    def _images_to_tensor(images):
        """
        Convert list of numpy.ndarray images with numpy.uint8 encoding ([0, 255] range)
            to torch.Tensor with torch.float32 encoding ([-1.0, 1.0] range)
        """
        images = np.stack(images)
        images = torch.from_numpy(np.transpose(images, axes=(0, 3, 1, 2)))
        return torch.clamp(images / 127.5 - 1.0, min=-1.0, max=1.0)

    @staticmethod
    def _calc_similarity(left_features, right_features):
        similarity_matrix = left_features @ right_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()

    @torch.no_grad()
    def _get_image_features(self, images, resolution=None):
        # noinspection PyPep8Naming
        PIL_images = [PIL.Image.fromarray(image) for image in images]
        dino_images_features = self.evaluator.get_dino_image_features(PIL_images)

        if resolution is not None:
            # noinspection PyTypeChecker,PyUnresolvedReferences
            images = [
                np.array(image.resize((resolution, resolution), resample=PIL.Image.BICUBIC))
                for image in PIL_images
            ]
        images = self._images_to_tensor(images)
        images_features = self.evaluator.get_image_features(images)

        return images_features, dino_images_features

    @torch.no_grad()
    def __call__(self, viewer: MultifolderViewer, config, *, verbose=False):
        results = {
            'image_similarities': {},
            'image_similarities_mx': {},

            'dino_image_similarities': {},
            'dino_image_similarities_mx': {},

            'text_similarities': {},
            'text_similarities_mx': {},

            'text_similarities_with_class': {},
            'text_similarities_mx_with_class': {},

            'text_similarities_prompt': {},
            'text_similarities_with_class_prompt': {},
        }
        train_data_dir, placeholder_token, class_name = get_prompt_info(config)

        train_image_paths, train_images = viewer.load_images(train_data_dir)

        train_images_features, dino_train_images_features = self._get_image_features(train_images, config['resolution'])

        results['real_image_similarity'], results['real_image_similarity_mx'] = (
            self._calc_similarity(train_images_features, train_images_features)
        )
        results['dino_real_image_similarity'], results['dino_real_image_similarity_mx'] = (
            self._calc_similarity(dino_train_images_features, dino_train_images_features)
        )
        if verbose:
            print('Real Image Similarity: {0:.3f} ({1:.3f})'.format(
                results['real_image_similarity'], results['dino_real_image_similarity']
            ))

        for label, images in viewer.images.items():
            images_features, dino_images_features = self._get_image_features(images)

            results['image_similarities'][label], results['image_similarities_mx'][label] = (
                self._calc_similarity(train_images_features, images_features)
            )
            results['dino_image_similarities'][label], results['dino_image_similarities_mx'][label] = (
                self._calc_similarity(dino_train_images_features, dino_images_features)
            )
            if verbose:
                print('IS: {0:.3f} ({1:.3f}) {2}'.format(
                    results['image_similarities'][label], results['dino_image_similarities'][label], label,
                ))

            clean_label = (
                label
                .replace('{0} {1}'.format(placeholder_token, class_name), '{0}')
                .replace('{0}'.format(placeholder_token), '{0}')
            )
            empty_label = clean_label.replace('{0} ', '').replace(' {0}', '')
            empty_label_with_class = clean_label.format(class_name)

            empty_label_features = self.evaluator.get_text_features(empty_label)
            empty_label_with_class_features = self.evaluator.get_text_features(empty_label_with_class)

            (
                results['text_similarities'][label],
                results['text_similarities_mx'][label]
            ) = self._calc_similarity(empty_label_features, images_features)
            results['text_similarities_prompt'][label] = empty_label

            (
                results['text_similarities_with_class'][label],
                results['text_similarities_mx_with_class'][label],
            ) = self._calc_similarity(empty_label_with_class_features, images_features)
            results['text_similarities_with_class_prompt'][label] = empty_label_with_class

            if verbose:
                print('TS: {0:.3f} {1}/{2}'.format(
                    results['text_similarities'][label], label,
                    results['text_similarities_prompt'][label]
                ))
                print('TS: {0:.3f} {1}/{2}'.format(
                    results['text_similarities_with_class'][label], label,
                    results['text_similarities_with_class_prompt'][label]
                ))

        return results


def narrow_similarities(
        text_similarities: Dict[str, float], prompts: List[str], holder: Optional[str], verbose: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """ Takes dictionary of similarities (prompt -> value) and aggregates only values from the list of prompts
    :param text_similarities: dictionary of text similarities
    :param prompts: subset of templated keys (i.e. "a photo of {0}") from dictionary of text similarities
    :param holder: target holder (i.e. placeholder or placeholder with class name)
    :param verbose: if True then print existing errors
    :return: mean and std text similarity over list of selected prompts. If some prompts are missing then returns None
    """
    try:
        result = []
        for prompt in prompts:
            result.append(text_similarities[prompt.format(holder)])

        return float(np.mean(result)), float(np.std(result))
    except Exception as ex:
        if verbose:
            print(f'Exception in narrow_similarities: {ex}')
        return None, None


def get_prompt_info(config):
    train_data_dir = config.get('test_data_dir', config.get('train_data_dir', config.get('instance_data_dir', None)))
    return train_data_dir, config['placeholder_token'], config['class_name']


def aggregate_similarities(data: dict) -> dict:
    """ Calculate necessary statistics over raw evaluation results
    :param dict data: output of ExpViewer._evaluate
    :return: dict with added aggregated stats
    """
    train_data_dir, placeholder_token, class_name = get_prompt_info(data['config'])

    with_attention_maps = data['config'].get('attention_masks_dir') is not None

    attention_masks_dir = data['config'].get('attention_masks_dir')
    if attention_masks_dir is not None:
        attention_map_mode = '/'.join(Path(attention_masks_dir).parts[-3:])
    else:
        attention_map_mode = None
    prompts_attention_map_mode = data['config'].get('prompts_attention_map_mode')

    result = {
        'with_attention_maps': with_attention_maps,
        'dataset': os.path.basename(train_data_dir),

        'attention_map_mode': attention_map_mode,
        'prompts_attention_map_mode': prompts_attention_map_mode,
    }

    base_holder, base_holder_with_class = placeholder_token, f'{placeholder_token} {class_name}'
    specs = []
    for set_name, set_prompts in evaluation_sets.items():
        base_metrics_names = [
            '_'.join([set_name, 'text_similarity']).lstrip('_'),
            '_'.join([set_name, 'image_similarity']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'text_similarity']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'image_similarity']).lstrip('_'),
        ]

        for metric_prefix in ['', 'dino_', 'masked_', 'masked_dino_']:
            if metric_prefix + 'image_similarities' in data:
                specs += [
                    (metric_prefix + base_metrics_names[1], metric_prefix + 'image_similarities', set_prompts, base_holder),
                    (metric_prefix + base_metrics_names[3], metric_prefix + 'image_similarities', set_prompts, base_holder_with_class)
                ]

        if 'text_similarities' in data:
            specs += [
                (base_metrics_names[0], 'text_similarities', set_prompts, base_holder),
                (base_metrics_names[2], 'text_similarities', set_prompts, base_holder_with_class),

                (base_metrics_names[0] + '_with_class', 'text_similarities_with_class', set_prompts, base_holder),
                (base_metrics_names[2] + '_with_class', 'text_similarities_with_class', set_prompts, base_holder_with_class),
            ]
    for key, similarities, prompts, holder in specs:
        result[key], result[key + '_std'] = narrow_similarities(
            data[similarities], prompts, holder, verbose=False
        )

    return result


def get_results_dataframe(all_data: Dict[str, dict]) -> pd.DataFrame:
    """ Gets dictionary with all results and combine them into DataFrame
    :param Dict[str, dict] all_data: dict of outputs of ExpViewer._evaluate
    :return: dataframe with all results
    """
    target_keys = [
        'dataset', 'with_class_name', 'with_attention_maps',

        'attention_map_mode', 'prompts_attention_map_mode',

        # Metrics for base prompt
        'image_similarity', 'with_class_image_similarity', 'real_image_similarity',
        'dino_image_similarity', 'dino_with_class_image_similarity', 'dino_real_image_similarity',

        # Metrics for medium evaluation set
        'image_similarity', 'medium_with_class_image_similarity',
        'dino_medium_image_similarity', 'medium_joined_with_class_image_similarity',

        'medium_text_similarity', 'medium_with_class_text_similarity',
        'medium_text_similarity_with_class', 'medium_with_class_text_similarity_with_class',
    ]

    results = defaultdict(list)
    for key, data in all_data.items():
        exp_name, (checkpoint_idx, num_inference_steps, guidance_scale, *other_inference_specs) = key
        results['exp_name'].append(exp_name)
        results['checkpoint_idx'].append(int(checkpoint_idx))
        results['num_inference_steps'].append(int(num_inference_steps))
        results['other_inference_specs'].append(other_inference_specs)
        results['guidance_scale'].append(float(guidance_scale))
        try:
            data |= aggregate_similarities(data)
            for target_key in target_keys:
                results[target_key].append(data.get(target_key))
                if 'similarity' in target_key:
                    results[target_key + '_std'].append(data.get(target_key + '_std'))

        except Exception as ex:
            print(exp_name, ex, traceback.format_exc())
            raise ex

    df = pd.DataFrame(data=results)

    return df
