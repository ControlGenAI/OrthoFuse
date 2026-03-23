import os
import gc
from typing import Callable, Any, Optional
import yaml
import random
import secrets
import logging
import itertools
from collections import defaultdict

from tqdm import tqdm
import wandb

import numpy as np

import torch
from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
)
from diffusers.loaders import AttnProcsLayers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
import transformers.utils.logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import sys
sys.path.append('/home/aalanov/vsoboleva/base_code/moft')

from nb_utils.eval_sets import small_set
from nb_utils.configs import live_object_data
from nb_utils.images_viewer import MultifolderViewer
from nb_utils.clip_eval import ExpEvaluator, aggregate_similarities

from .utils.registry import ClassRegistry
from .model.moft import MOFTCrossAttnProcessor
from .model.lora import LoRACrossAttnProcessor
from .model.utils_sdxl import count_trainable_params, params_grad_norm, cast_training_params
from .data.dataset_sdxl import ImageDataset, DreamBoothDataset, StyleDataset, collate_fn, tokenize_prompt, encode_tokens, compute_time_ids

logger = get_logger(__name__)
trainers = ClassRegistry()

BASE_PROMPT = "a photo of {0}"

torch.backends.cuda.enable_flash_sdp(True)


@trainers.add_to_registry('sdxl_concept')
class ConceptTrainerSDXL:

    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(dict)

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.train_data_dir))
        )
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Get the last experiment idx
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            # noinspection PyBroadException
            try:
                curr_exp_idx = max(exp_idx, int(folder.split('-')[0].lstrip('0')))
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        self.config.exp_name = self.setup_exp_name(exp_idx)

        self.config.output_dir = os.path.abspath(os.path.join(self.config.output_dir, self.config.exp_name))

        if os.path.exists(self.config.output_dir):
            raise ValueError(f'Experiment directory {self.config.output_dir} already exists. Race condition!')
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.config.logging_dir = os.path.join(self.config.output_dir, 'logs')
        os.makedirs(self.config.logging_dir, exist_ok=True)

        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        if self.config.api_key is not None:
            wandb.login(key=self.config.api_key)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir)
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            #log_with='wandb',
            project_config=accelerator_project_config,
        )
        self.accelerator.init_trackers(
            project_name=self.config.project_name,
            config=self.config,
            init_kwargs={"wandb": {
                "name": self.config.exp_name,
                'settings': wandb.Settings(code_dir=os.path.dirname(self.config.argv[1]))
            }}
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler", revision=self.config.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet", revision=self.config.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder", revision=self.config.revision
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2", revision=self.config.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix',
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer", revision=self.config.revision
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2", revision=self.config.revision
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        self.params_to_optimize = []

        moft_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            moft_attn_procs[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config.moft_nblocks,
                method=self.config.moft_method, scale=self.config.moft_scale
            )

        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.moft_layers)

        for (name, param) in self.moft_layers.named_parameters():
            param.requires_grad = True
            self.params_to_optimize.append(param)
        self.moft_layers.train()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-04,
            eps=1e-08,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt=BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}'),
                class_data_root=self.config.class_data_dir if self.config.with_prior_preservation else None,
                class_prompt=BASE_PROMPT.format(self.config.class_name),
                tokenizers=(self.tokenizer, self.tokenizer_2),
                size=self.config.resolution
            )
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
                examples, self.config.with_prior_preservation
            )
        else:
            self.train_dataset = ImageDataset(
                train_data_dir=self.config.train_data_dir,
                resolution=self.config.resolution
            )
            collator = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator
        )

    # noinspection PyTypeChecker
    def move_to_device(self):
        self.optimizer, self.train_dataloader = self.accelerator.prepare(self.optimizer, self.train_dataloader)
        self.moft_layers = self.accelerator.prepare(self.moft_layers)

        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        # # All trained parameters should be explicitly moved to float32 even for mixed precision training
        cast_training_params((self.unet, self.text_encoder, self.text_encoder_2), dtype=torch.float32)

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(batch['pixel_values'].to(self.weight_dtype)).latent_dist.sample()
        else:
            latents = self.vae.encode(batch['image'].to(self.weight_dtype) * 2.0 - 1.0).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation:
            input_ids_list = tokenize_prompt(
                (self.tokenizer, self.tokenizer_2),
                BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}')
            )
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), input_ids_list
            )
        else:
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), (batch["input_ids"], batch["input_ids_2"])
            )

        # Note: encoder_hidden_states contains embeddings for the following combinations of tokens:
        # if with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'CLASS_NAME</w>',                          '<|endoftext|>', ...PADDING]
        # if not with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states
        }
        print(noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape)
        print(unet_added_conditions['time_ids'].shape, unet_added_conditions['text_embeds'].shape)
        outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(prior_outputs.float(), prior_target.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder, keep_fp32_wrapper=False),
            text_encoder_2=self.accelerator.unwrap_model(self.text_encoder_2, keep_fp32_wrapper=False),
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=self.weight_dtype if self.accelerator.mixed_precision in ["fp16", 'bf16'] else torch.float16
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    # noinspection PyPep8Naming
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = small_set

        samples_path = os.path.join(
            self.config.output_dir, f'checkpoint-{epoch}', 'samples',
            'ns0_gs0_validation', 'version_0'
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(f'{self.config.placeholder_token} {self.config.class_name}')
                kwargs = {
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f'{idx}.png'))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    'validation': [
                        wandb.Image(image, caption=caption) for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)

        self.unet.save_attn_procs(os.path.join(save_path))
        #self.unet.save_pretrained(os.path.join(save_path, "saved_adapter"), save_adapter=True)

    def train(self):
        if self.accelerator.is_main_process:
            # self.validation(0)
            # self.save_model(0)
            print(1)

        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            with self.accelerator.autocast():
                loss = self.train_step(batch)
            self.accelerator.backward(loss)

            for tracker in self.accelerator.trackers:
                tracker.log({
                    "loss": loss,
                    "unet_grad_norm": params_grad_norm(self.unet.parameters()),
                    "text_encoder_grad_norm": params_grad_norm(self.text_encoder.parameters()),
                    "text_encoder_2_grad_norm": params_grad_norm(self.text_encoder_2.parameters())
                })

            self.optimizer.step()
            self.optimizer.zero_grad()

            del batch, loss
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch != 0:
                    self.validation(epoch)
                    self.save_model(epoch)

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            self.validation(self.config.num_train_epochs)
            self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()

    def setup_evaluator(self):
        self.evaluator = ExpEvaluator(self.accelerator.device)

    def _evaluate(self, path):
        viewer = MultifolderViewer(path, lazy_load=False)

        results = self.evaluator(viewer, vars(self.config))
        results |= {'config': vars(self.config)}
        results |= aggregate_similarities(results)

        return results


style_small_set = [
    'a dog in {0}',
    'a cat in {0}',
    'an sea view with palms in {0}',
    'meadow with flowers in {0}',
    'a beautiful blonde girl in {0}'
]
@trainers.add_to_registry('sdxl_style')
class StyleTrainerSDXL(ConceptTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_dataset(self):
        print("train_data_dir:", self.config.train_data_dir)
        self.train_dataset = StyleDataset(
            instance_image_root=self.config.train_data_dir,
            tokenizers=(self.tokenizer, self.tokenizer_2),
            placeholder_token=self.config.placeholder_token,
            class_name=self.config.class_name,
            size=self.config.resolution
        )
        collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
            examples, self.config.with_prior_preservation
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator
        )

    # noinspection PyPep8Naming
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = style_small_set

        samples_path = os.path.join(
            self.config.output_dir, f'checkpoint-{epoch}', 'samples',
            'ns0_gs0_validation', 'version_0'
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(f'{self.config.placeholder_token} {self.config.class_name}')
                kwargs = {
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f'{idx}.png'))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    'validation': [
                        wandb.Image(image, caption=caption) for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def train_step(self, batch):
        latents = self.vae.encode(batch['pixel_values'].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        input_ids_list = tokenize_prompt(
            (self.tokenizer, self.tokenizer_2),
            self.train_dataset.instance_prompt
        )
        encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
            (self.text_encoder, self.text_encoder_2), input_ids_list
        )

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states
        }
        outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions).sample

        loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss


@trainers.add_to_registry('sdxl_style_lora')
class StyleTrainerLoraSDXL(StyleTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        self.params_to_optimize = []

        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config.lora_rank)
                #print("rank:", rank)
            else:
                rank = min(hidden_size, self.config.lora_rank)
                #print("rank:", rank)
            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.lora_layers)

        for (name, param) in self.lora_layers.named_parameters():
            param.requires_grad = True
            self.params_to_optimize.append(param)
        self.lora_layers.train()


@trainers.add_to_registry('sdxl_concept_lora')
class ConceptTrainerLoraSDXL:

    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(dict)

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.train_data_dir))
        )
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Get the last experiment idx
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            # noinspection PyBroadException
            try:
                curr_exp_idx = max(exp_idx, int(folder.split('-')[0].lstrip('0')))
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        self.config.exp_name = self.setup_exp_name(exp_idx)

        self.config.output_dir = os.path.abspath(os.path.join(self.config.output_dir, self.config.exp_name))

        if os.path.exists(self.config.output_dir):
            raise ValueError(f'Experiment directory {self.config.output_dir} already exists. Race condition!')
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.config.logging_dir = os.path.join(self.config.output_dir, 'logs')
        os.makedirs(self.config.logging_dir, exist_ok=True)

        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        if self.config.api_key is not None:
            wandb.login(key=self.config.api_key)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir)
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            #log_with='wandb',
            project_config=accelerator_project_config,
        )
        self.accelerator.init_trackers(
            project_name=self.config.project_name,
            config=self.config,
            init_kwargs={"wandb": {
                "name": self.config.exp_name,
                'settings': wandb.Settings(code_dir=os.path.dirname(self.config.argv[1]))
            }}
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler", revision=self.config.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet", revision=self.config.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder", revision=self.config.revision
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2", revision=self.config.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix',
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer", revision=self.config.revision
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2", revision=self.config.revision
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        self.params_to_optimize = []

        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config.lora_rank)
                #print("rank:", rank)
            else:
                rank = min(hidden_size, self.config.lora_rank)
                #print("rank:", rank)
            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.lora_layers)

        for (name, param) in self.lora_layers.named_parameters():
            param.requires_grad = True
            self.params_to_optimize.append(param)
        self.lora_layers.train()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-04,
            eps=1e-08,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt=BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}'),
                class_data_root=self.config.class_data_dir if self.config.with_prior_preservation else None,
                class_prompt=BASE_PROMPT.format(self.config.class_name),
                tokenizers=(self.tokenizer, self.tokenizer_2),
                size=self.config.resolution
            )
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
                examples, self.config.with_prior_preservation
            )
        else:
            self.train_dataset = ImageDataset(
                train_data_dir=self.config.train_data_dir,
                resolution=self.config.resolution
            )
            collator = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator
        )

    # noinspection PyTypeChecker
    def move_to_device(self):
        self.optimizer, self.train_dataloader = self.accelerator.prepare(self.optimizer, self.train_dataloader)
        self.lora_layers = self.accelerator.prepare(self.lora_layers)

        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        # # All trained parameters should be explicitly moved to float32 even for mixed precision training
        cast_training_params((self.unet, self.text_encoder, self.text_encoder_2), dtype=torch.float32)

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(batch['pixel_values'].to(self.weight_dtype)).latent_dist.sample()
        else:
            latents = self.vae.encode(batch['image'].to(self.weight_dtype) * 2.0 - 1.0).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation:
            input_ids_list = tokenize_prompt(
                (self.tokenizer, self.tokenizer_2),
                BASE_PROMPT.format(f'{self.config.placeholder_token} {self.config.class_name}')
            )
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), input_ids_list
            )
        else:
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), (batch["input_ids"], batch["input_ids_2"])
            )

        # Note: encoder_hidden_states contains embeddings for the following combinations of tokens:
        # if with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'CLASS_NAME</w>',                          '<|endoftext|>', ...PADDING]
        # if not with_prior_preservation:
        #   ['<|startoftext|>', 'a</w>', 'photo</w>', 'of</w>', 'PLACEHOLDER_TOKEN</w>', 'CLASS_NAME</w>', '<|endoftext|>', ...PADDING]

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states
        }
        print(noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape)
        print(unet_added_conditions['time_ids'].shape, unet_added_conditions['text_embeds'].shape)
        outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(prior_outputs.float(), prior_target.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder, keep_fp32_wrapper=False),
            text_encoder_2=self.accelerator.unwrap_model(self.text_encoder_2, keep_fp32_wrapper=False),
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=self.weight_dtype if self.accelerator.mixed_precision in ["fp16", 'bf16'] else torch.float16
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    # noinspection PyPep8Naming
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = small_set

        samples_path = os.path.join(
            self.config.output_dir, f'checkpoint-{epoch}', 'samples',
            'ns0_gs0_validation', 'version_0'
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(f'{self.config.placeholder_token} {self.config.class_name}')
                kwargs = {
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f'{idx}.png'))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    'validation': [
                        wandb.Image(image, caption=caption) for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)

        self.unet.save_attn_procs(os.path.join(save_path))
        #self.unet.save_pretrained(os.path.join(save_path, "saved_adapter"), save_adapter=True)

    def train(self):
        if self.accelerator.is_main_process:
            # self.validation(0)
            # self.save_model(0)
            print(1)

        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            with self.accelerator.autocast():
                loss = self.train_step(batch)
            self.accelerator.backward(loss)

            for tracker in self.accelerator.trackers:
                tracker.log({
                    "loss": loss,
                    "unet_grad_norm": params_grad_norm(self.unet.parameters()),
                    "text_encoder_grad_norm": params_grad_norm(self.text_encoder.parameters()),
                    "text_encoder_2_grad_norm": params_grad_norm(self.text_encoder_2.parameters())
                })

            self.optimizer.step()
            self.optimizer.zero_grad()

            del batch, loss
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch != 0:
                    self.validation(epoch)
                    self.save_model(epoch)

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            self.validation(self.config.num_train_epochs)
            self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()

    def setup_evaluator(self):
        self.evaluator = ExpEvaluator(self.accelerator.device)

    def _evaluate(self, path):
        viewer = MultifolderViewer(path, lazy_load=False)

        results = self.evaluator(viewer, vars(self.config))
        results |= {'config': vars(self.config)}
        results |= aggregate_similarities(results)

        return results
    
    
@trainers.add_to_registry('sdxl_style_lora')
class StyleTrainerLoraSDXL(ConceptTrainerLoraSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_dataset(self):
        self.train_dataset = StyleDataset(
            instance_image_root=self.config.train_data_dir,
            tokenizers=(self.tokenizer, self.tokenizer_2),
            placeholder_token=self.config.placeholder_token,
            class_name=self.config.class_name,
            size=self.config.resolution
        )
        collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
            examples, self.config.with_prior_preservation
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator
        )

    # noinspection PyPep8Naming
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = style_small_set

        samples_path = os.path.join(
            self.config.output_dir, f'checkpoint-{epoch}', 'samples',
            'ns0_gs0_validation', 'version_0'
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(f'{self.config.placeholder_token} {self.config.class_name}')
                kwargs = {
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f'{idx}.png'))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    'validation': [
                        wandb.Image(image, caption=caption) for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def train_step(self, batch):
        latents = self.vae.encode(batch['pixel_values'].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        input_ids_list = tokenize_prompt(
            (self.tokenizer, self.tokenizer_2),
            self.train_dataset.instance_prompt
        )
        encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
            (self.text_encoder, self.text_encoder_2), input_ids_list
        )

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states
        }
        outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_conditions).sample

        loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss
