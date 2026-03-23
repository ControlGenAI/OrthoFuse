import os
from tqdm import tqdm
import random
import numpy as np
from functools import reduce
import torch
from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
)
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from peft import  LoraConfig, get_peft_model
from .model.moft import MOFTCrossAttnProcessor, DoubleMOFTCrossAttnProcessor, MOFTDoubleCrossAttnProcessor
from .model.lora import LoRACrossAttnProcessor
from .utils.registry import ClassRegistry
from .utils.gs_orthogonal import merge_inside_cayley_space, blocked_geodesic_combination, full_matrix_geodesic_combination, GSOrthogonal, postprocess_blocks, merge_inside_cayley_space_batch, postprocess_blocks_batch
from .model.monarch_orthogonal import MonarchOrthogonal
from .utils.fixed_rank_batch import FixedRankBatch
from diffusers import EulerDiscreteScheduler
import time
import json 

inferencers = ClassRegistry()


def get_seed(prompt, i, seed):
    h = 0
    for el in prompt:
        h += ord(el)
    h += i
    return h + seed


@inferencers.add_to_registry('base')
class BaseInferencer:
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):
        print("BaseInferencer init")
        self.config = config
        self.args = args
        self.checkpoint_idx = args.checkpoint_idx
        self.num_images_per_context_prompt = args.num_images_per_medium_prompt
        self.num_images_per_base_prompt = args.num_images_per_base_prompt
        self.batch_size_context = args.batch_size_medium
        self.batch_size_base = args.batch_size_base

        if self.checkpoint_idx is None:
            self.checkpoint_path = config['output_dir']
        else:
            self.checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{self.checkpoint_idx}')

        self.context_prompts = context_prompts
        self.base_prompts = base_prompts

        self.replace_inference_output = self.args.replace_inference_output
        self.version = self.args.version

        self.device = device
        self.dtype = dtype

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'num_inference_steps': self.args.num_inference_steps,
        }

    def setup_base_model(self):
        # Here we create base models
        # self.scheduler = DDIMScheduler.from_pretrained(
        #     self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        # )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )

    def setup_model(self):
        self.unet.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'unet.bin')
        ))

    def setup_pipeline(self):
        print("setup_pipeline for SDXL")
        #self.pipe = StableDiffusionPipeline.from_pretrained(
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            # scheduler=self.scheduler,
            # tokenizer=self.tokenizer,
            unet=self.unet,
            # text_encoder=self.text_encoder,
            revision=None,
            requires_safety_checker=False,
            torch_dtype=self.dtype,
            # local_files_only=True
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def setup(self):
        self.setup_base_model()
        self.setup_model()
        self.setup_pipeline()
        self.setup_pipe_kwargs()
        self.create_folder_name()
        self.setup_paths()

    def prepare_prompts(self, context_prompts, base_prompts):
        return context_prompts, base_prompts

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"

    def setup_paths(self):
        if self.version is None:
            version = 0
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
            )
            print("samples_path", samples_path)
            if os.path.exists(samples_path):
                while not os.path.exists(samples_path):
                    samples_path = os.path.join(
                        self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
                    )
                    version += 1
        else:
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{self.version}'
            )
        self.samples_path = samples_path

    def check_generation(self, path, num_images_per_prompt):
        if self.replace_inference_output:
            return True
        else:
            if os.path.exists(path) and len(os.listdir(path)) == num_images_per_prompt:
                return False
            else:
                return True

    def generate_with_prompt(self, prompt, num_images_per_prompt, batch_size):
        n_batches = (num_images_per_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            seed = get_seed(prompt, i, self.args.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            generator = torch.Generator(device='cuda')
            generator = generator.manual_seed(seed)
            print("Prompt before formatting:", prompt)
            print("Prompt after formatting:", prompt.format(f"{self.config['placeholder_token_concept']} {self.config['class_name']}", self.config['placeholder_token_style']))
            
            images_batch = self.pipe(
                prompt=prompt.format(f"{self.config['placeholder_token_concept']} {self.config['class_name']}", self.config['placeholder_token_style']),
                generator=generator, 
                num_images_per_prompt=batch_size,
                added_cond_kwargs={}, 
                **self.pipe_kwargs
            ).images
            images += images_batch
        return images

    def save_images(self, images, path):
        os.makedirs(path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f'{idx}.png'))

    def generate_with_prompt_list(self, prompts, num_images_per_prompt, batch_size):
        print('num_images_per_prompt:',num_images_per_prompt)
        for prompt in tqdm(prompts):
            formatted_prompt = prompt.format(self.config['placeholder_token_concept'], self.config['placeholder_token_style'])
            path = os.path.join(self.samples_path, formatted_prompt)
            if self.check_generation(path, num_images_per_prompt):
                images = self.generate_with_prompt(
                    prompt, num_images_per_prompt, batch_size)
                self.save_images(images, path)

    def generate(self):
        context_prompts, base_prompts = self.prepare_prompts(self.context_prompts, self.base_prompts)
        self.generate_with_prompt_list(
            context_prompts, self.num_images_per_context_prompt, self.batch_size_context)


@inferencers.add_to_registry('lora')
class LoraInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
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
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )

        self.unet.set_attn_processor(lora_attn_procs)
        self.unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.unet_lora_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))


@inferencers.add_to_registry('moft')
class MOFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
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
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )

        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))


@inferencers.add_to_registry('moft_merge')
class MOFTMergeInferencer(BaseInferencer):
    """
    MOFTMergeInferencer performs merge of orthogonal adapters and than runs
    inference. The merge in particular is implemented in layer_merge function.
    layer_merge supports two types of merging: block-wise merging and merging
    inside Cayley space, which corresponds to weighted sum of skew-hermittian matrix.
    """
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
     
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}_method_{self.args.postprocessing_method}" 
    
    def layer_merge(self, t, layer = "to_q_moft", merging_type="blocked", parameter=1, postprocessing_method="uniform"):
        for name, proc1 in self.unet.attn_processors.items():
            proc2 = self.unet_style.attn_processors[name]
            if hasattr(proc1, layer) and hasattr(proc2, layer):
                if merging_type == "blocked":
                    merged_gs = blocked_geodesic_combination(
                            getattr(proc1, layer).ort_monarch,
                            getattr(proc2, layer).ort_monarch,
                            t
                    )

                    assert merged_gs.method == "already_orthogonal", "merge is not already_orthogonal" # Remove it after debugging
                elif merging_type == "merge_inside_cayley_space":
                    merged_gs = merge_inside_cayley_space(
                            [getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch],
                            torch.tensor([t, 1-t], device=getattr(proc1, layer).ort_monarch.L.data.device)
                    )
                    
                    assert merged_gs.method == "already_orthogonal", "merge is not already_orthogonal" # Remove it after debugging

                # Here we modify eigenvalues of the already orthogonal matrix
                merged_gs = postprocess_blocks(merged_gs, method=postprocessing_method, parameter=parameter)
                getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                getattr(proc1, layer).ort_monarch.method = "already_orthogonal"         

    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model SDXL ...")
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        ).to(self.device)
        self.unet_style = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )

    def setup_model(self,):
        print("setup_model ...")
        print("self.device:", self.device)
        moft_attn_procs = {}
        moft_attn_procs_style = {}
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
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            moft_attn_procs_style[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        # self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_concept.safetensors')))
        if self.args.moft_layers_concept_path is not None:
            self.moft_layers.load_state_dict(load_file(self.args.moft_layers_concept_path))
        
        else:
            raise ValueError("moft_layers_concept_path is not provided")
        
        self.unet_style.set_attn_processor(moft_attn_procs_style)
        self.moft_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        if self.args.moft_layers_style_path is not None:
            self.moft_layers_style.load_state_dict(load_file(self.args.moft_layers_style_path))
        else:
            raise ValueError("moft_layers_style_path is not provided")

        # Перенос всех весов на device
        for proc in self.unet.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)
        for proc in self.unet_style.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)

        if self.args.t is not None:
            t = self.args.t
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge
        # merging_type = "merge_inside_cayley_space"
        merging_type = "blocked"
        merging = True
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", t)
            print("merging to_q_moft...")
            self.layer_merge(t, layer = "to_q_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_k_moft...")
            self.layer_merge(t, layer = "to_k_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_v_moft...")
            self.layer_merge(t, layer = "to_v_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging to_out_moft...")
            self.layer_merge(t, layer = "to_out_moft", merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging done!")


@inferencers.add_to_registry('lora_merge')
class LoraMergeInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
    
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}" 
    
    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model ...")
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.unet_style = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )
        
    def setup_model(self,):
        lora_attn_procs = {}
        lora_attn_procs_style = {}
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
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config['lora_rank'])
            else:
                rank = min(hidden_size, self.config['lora_rank'])

            lora_attn_procs_style[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank,
            )
            
        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.lora_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_concept.safetensors')))
        
        self.unet_style.set_attn_processor(lora_attn_procs_style)
        self.lora_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        self.lora_layers_style.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_style.safetensors')))

        if self.args.t is not None:
            t = self.args.t
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge

        merging_type = "frb"
        merging = True
        
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", t)
            print("merging to_q_lora...")
            self.layer_merge(t, layer = "to_q_lora", merging_type=merging_type)
            print("merging to_k_lora...")
            self.layer_merge(t, layer = "to_k_lora", merging_type=merging_type)
            print("merging to_v_lora...")
            self.layer_merge(t, layer = "to_v_lora", merging_type=merging_type)
            print("merging to_out_lora...")
            self.layer_merge(t, layer = "to_out_lora", merging_type=merging_type)
            print("merging done!")
        print("Merged L norm:", self.unet.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_lora.down.weight.norm())
         
    def layer_merge(self, t, layer = "to_q_lora", merging_type="blocked"):
        # Create a directory for saving matrices if it doesn't exist
        matrices_dir = os.path.join(self.checkpoint_path, 'matrices')
        os.makedirs(matrices_dir, exist_ok=True)
        
        for name, proc1 in self.unet.attn_processors.items():
            proc2 = self.unet_style.attn_processors[name]
            if hasattr(proc1, layer) and hasattr(proc2, layer):
                                   
                if merging_type == "frb":
                    if hasattr(getattr(proc1, layer), 'down'):
                        A_proc1 = getattr(proc1, layer).down.weight
                        B_proc1 = getattr(proc1, layer).up.weight
                        A_proc2 = getattr(proc2, layer).down.weight
                        B_proc2 = getattr(proc2, layer).up.weight
                        fr_batch = [
                            (A_proc1.T, B_proc1),
                            (A_proc2.T, B_proc2),
                        ] 

                        conv_coef = torch.tensor([t, 1-t])

                        frb = FixedRankBatch(fr_batch, conv_coef)
                        approx = frb.riemannian_barycenter_approximation()
                        new_matrix = approx.to_dense()
                        with torch.no_grad():
                            getattr(proc1, layer).down.weight.copy_(approx.U.T)
                            getattr(proc1, layer).up.weight.copy_(approx.V)
 
                else:
                    raise ValueError(f"Invalid merging type: {merging_type}")


@inferencers.add_to_registry('moft_merge_fast')
class MOFTMergeFastInferencer(BaseInferencer):
    """
    MOFTMergeFastInferencer is the accelerated version of MOFTMergeInferencer. 
    """
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
     
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}_method_{self.args.postprocessing_method}_ultra_fast" 
    
    def layer_merge(self, t, layers = ["to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft"], merging_type="blocked", parameter=1, postprocessing_method="uniform"):
      
        filename = "layers_distribution.json"
        with open(filename, 'r') as f:
            layers_distribution = json.load(f)

        # Локальные ссылки на словари процессоров, чтобы не дергать
        # атрибуты self.unet / self.unet_style в каждом шаге цикла.
        
        unet_attn_procs = self.unet.attn_processors
        unet_style_attn_procs = self.unet_style.attn_processors
        layer_size_list = [20, 40, 64]   # 3 размера слоев

        for layer_size in layer_size_list:
            layer_q_moft = layers_distribution['to_q_moft'][str(layer_size)] # берем названия слоев to_q_moft с конкретным размером
            layer_k_moft = layers_distribution['to_k_moft'][str(layer_size)]
            layer_v_moft = layers_distribution['to_v_moft'][str(layer_size)]
            layer_out_moft = layers_distribution['to_out_moft'][str(layer_size)]
            p1_list, p2_list = [], []
            for layer in layer_q_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_q_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_q_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)
            
            for layer in layer_k_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_k_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_k_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            for layer in layer_v_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_v_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_v_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            for layer in layer_out_moft:
                proc1 = unet_attn_procs[layer]
                proc2 = unet_style_attn_procs[layer]
                p1_monarch = getattr(proc1, "to_out_moft").ort_monarch
                p2_monarch = getattr(proc2, "to_out_moft").ort_monarch
                p1_list.append(p1_monarch)
                p2_list.append(p2_monarch)

            L1 = torch.stack([p.L.data for p in p1_list], dim=0)  # [B, nblocks, d, d]
            R1 = torch.stack([p.R.data for p in p1_list], dim=0)
            L2 = torch.stack([p.L.data for p in p2_list], dim=0)
            R2 = torch.stack([p.R.data for p in p2_list], dim=0)
            L_ortho, R_ortho = merge_inside_cayley_space_batch(
                L1,
                R1,
                L2,
                R2,
                torch.tensor([t, 1-t], device=p1_list[0].L.data.device)
            )

            # batch post-processing of orthogonal matrices
            L_ortho, R_ortho = postprocess_blocks_batch(
                L_ortho, R_ortho, method=postprocessing_method, parameter=parameter
            )

            # write back results into p1_list
            B = L_ortho.shape[0]
            assert B == len(p1_list), "Batch size mismatch in layer_merge"
            for i in range(B):
                p1_list[i].L.data.copy_(L_ortho[i])
                p1_list[i].R.data.copy_(R_ortho[i])
                p1_list[i].method = "already_orthogonal"

    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model SDXL ...")
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        ).to(self.device)
        self.unet_style = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )

    def setup_model(self,):
        print("setup_model ...")
        print("self.device:", self.device)
        moft_attn_procs = {}
        moft_attn_procs_style = {}
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
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
            
        for name in self.unet_style.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_style.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet_style.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_style.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_style.config.block_out_channels[block_id]

            moft_attn_procs_style[name] = MOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )
        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        if self.args.moft_layers_concept_path is not None:
            self.moft_layers.load_state_dict(load_file(self.args.moft_layers_concept_path))
            print("moft_layers_concept_path loaded", self.args.moft_layers_concept_path)
        else:
            raise ValueError("moft_layers_concept_path is not provided")
        
        self.unet_style.set_attn_processor(moft_attn_procs_style)
        self.moft_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        if self.args.moft_layers_style_path is not None:
            self.moft_layers_style.load_state_dict(load_file(self.args.moft_layers_style_path))
            print("moft_layers_style_path loaded", self.args.moft_layers_style_path)
        else:
            raise ValueError("moft_layers_style_path is not provided")

        # Перенос всех весов на device
        for proc in self.unet.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)
        for proc in self.unet_style.attn_processors.values():
            if hasattr(proc, "ort_monarch"):
                proc.ort_monarch = proc.ort_monarch.to(self.device)

        if self.args.t is not None:
            t = 1 - self.args.t # так сделано потому что в MOFTMergeInferencer t=0 это адаптер концепта, а t=1 это адаптер стиля, а в MOFTMergeFastInferencer порядок противоположный
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge
        merging_type = "merge_inside_cayley_space"
        # merging_type = "blocked"
        merging = True
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", 1-t)
            print("merging...")
            self.layer_merge(t, layers = ["to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft"], merging_type=merging_type, parameter=t, postprocessing_method=self.args.postprocessing_method)
            print("merging done!")
