import os
from tqdm import tqdm
import random
import numpy as np

import torch
from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
)
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from peft import  LoraConfig, get_peft_model
#from peft import BOFTConfig
from .model.moft import MOFTCrossAttnProcessor, DoubleMOFTCrossAttnProcessor
from .model.lora import LoRACrossAttnProcessor
from .utils.registry import ClassRegistry
from .utils.gs_orthogonal import merge_inside_cayley_space, blocked_geodesic_combination, full_matrix_geodesic_combination, GSOrthogonal, riemannian_barycenter_approximation, fast_riemannian_barycenter_approximation, fast_riemannian_barycenter_approximation_without_svd, riemannian_barycenter_approximation2_0, postprocess_blocks
from .model.monarch_orthogonal import MonarchOrthogonal
from .utils.fixed_rank_batch import FixedRankBatch

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
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
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
        #self.pipe = StableDiffusionPipeline.from_pretrained(
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            revision=None,
            requires_safety_checker=False,
            torch_dtype=self.dtype,
            local_files_only=True
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
            print("I am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            images_batch = self.pipe(
                prompt=prompt.format(f"{self.config['placeholder_token']} {self.config['class_name']}"),
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
            formatted_prompt = prompt.format(self.config['placeholder_token'])
            path = os.path.join(self.samples_path, formatted_prompt)
            if self.check_generation(path, num_images_per_prompt):
                images = self.generate_with_prompt(
                    prompt, num_images_per_prompt, batch_size)
                self.save_images(images, path)

    def generate(self):
        context_prompts, base_prompts = self.prepare_prompts(self.context_prompts, self.base_prompts)
        self.generate_with_prompt_list(
            context_prompts, self.num_images_per_context_prompt, self.batch_size_context)
        self.generate_with_prompt_list(
            base_prompts, self.num_images_per_base_prompt, self.batch_size_base)


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


@inferencers.add_to_registry('double_moft')
class DoubleMOFTInferencer(BaseInferencer):
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

            moft_attn_procs[name] = DoubleMOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['moft_nblocks'],
                method=self.config['moft_method'], scale=self.config['moft_scale']
            )

        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))


@inferencers.add_to_registry('boft')
class BOFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
        config = BOFTConfig(
            boft_block_size=0,
            boft_block_num=self.config['boft_block_num'],
            boft_n_butterfly_factor=self.config['boft_n_butterfly_factor'],
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            boft_dropout=0.0,
            bias=self.config['boft_bias'],
            inference_mode=True,
        )
        self.unet = get_peft_model(self.unet, config)
        self.unet.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'unet.bin')
        ))


@inferencers.add_to_registry('dora')
class DoraInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
        UNET_TARGET_MODULES = ["to_q", "to_v", "to_k", "to_out.0"]
        unet_lora_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_rank'],
            init_lora_weights="gaussian",
            target_modules=UNET_TARGET_MODULES,
            use_dora=True,
        )
        self.unet = get_peft_model(self.unet, unet_lora_config)
        self.unet.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'unet.bin')
        ))


@inferencers.add_to_registry('superclass_ft')
class SuperclassFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):
        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def create_folder_name(self):
        self.inference_folder_name = f'ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_superclassft'

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
            images_batch = self.pipe(
                prompt.format(self.config['class_name']), num_images_per_prompt=batch_size,
                generator=generator,
                **self.pipe_kwargs
            ).images
            images += images_batch
        return images


@inferencers.add_to_registry('superclass')
class SuperclassInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):
        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self):
        pass

    def create_folder_name(self):
        self.inference_folder_name = f'ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_superclass'

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
            images_batch = self.pipe(
                prompt.format(self.config['class_name']), num_images_per_prompt=batch_size,
                generator=generator,
                **self.pipe_kwargs
            ).images
            images += images_batch
        return images



@inferencers.add_to_registry('moft_merge')
class MOFTMergeInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
     
    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}_eta{self.args.eta}" 
    
    def layer_merge(self, t, layer = "to_q_moft", independent_merge=False, merging_type="blocked", eta=None):
        # Create a directory for saving matrices if it doesn't exist
        matrices_dir = os.path.join(self.checkpoint_path, 'matrices')
        os.makedirs(matrices_dir, exist_ok=True)
        count = 0
        for name, proc1 in self.unet.attn_processors.items():
            proc2 = self.unet_style.attn_processors[name]
            if hasattr(proc1, layer) and hasattr(proc2, layer):
                # Save matrices before merging
                #monarch_concept = getattr(proc1, layer).ort_monarch
                #monarch_style = getattr(proc2, layer).ort_monarch
                #torch.save(monarch_concept.state_dict(), os.path.join(matrices_dir, f'{name}_{layer}_monarch_concept_before.pt'))
                #torch.save(monarch_style.state_dict(), os.path.join(matrices_dir, f'{name}_{layer}_monarch_style_before.pt'))
                #print("name:", name)
                if independent_merge:
                    # Merge L
                    L1 = getattr(proc1, layer).ort_monarch.L.data
                    L2 = getattr(proc2, layer).ort_monarch.L.data
                    # L1 = proc1[layer].ort_monarch.L.data
                    # L2 = proc2[layer].ort_monarch.L.data
                    nblocks, n1, n2 = L1.shape   #[nblocks, block_size, block_size] - так как матрицы блочные
                    n = n1 * nblocks
                    g1 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
                    g2 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
                    g1.gsoft_L.data = L1.clone()
                    g2.gsoft_L.data = L2.clone()
                    g1.gsoft_R.data.zero_()
                    g2.gsoft_R.data.zero_()
                    merged_gs = blocked_geodesic_combination(g1, g2, t)
                    #proc1[layer].ort_monarch.L.data.copy_(merged_gs.gsoft_L.data)
                    getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.gsoft_L.data)
                    
                    # Merge R
                    # R1 = proc1[layer].ort_monarch.R.data
                    # R2 = proc2[layer].ort_monarch.R.data
                    R1 = getattr(proc1, layer).ort_monarch.R.data
                    R2 = getattr(proc2, layer).ort_monarch.R.data
                    g1.gsoft_R.data = R1.clone()
                    g2.gsoft_R.data = R2.clone()
                    g1.gsoft_L.data.zero_()
                    g2.gsoft_L.data.zero_()
                    merged_gs = blocked_geodesic_combination(g1, g2, t)
                    #proc1[layer].ort_monarch.R.data.copy_(merged_gs.gsoft_R.data)
                    getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.gsoft_R.data)
                    
                    # Save matrix after independent merging
                    #monarch_merged = getattr(proc1, layer).ort_monarch
                    #torch.save(monarch_merged.state_dict(), os.path.join(matrices_dir, f'{name}_{layer}_monarch_after_independent.pt'))
                
                else:
                    # L1 = getattr(proc1, layer).ort_monarch.L.data
                    # L2 = getattr(proc2, layer).ort_monarch.L.data
                    # R1 = getattr(proc1, layer).ort_monarch.R.data
                    # R2 = getattr(proc2, layer).ort_monarch.R.data
                    
                    # nblocks, n1, n2 = L1.shape   #[nblocks, block_size, block_size] - так как матрицы блочные
                    # n = n1 * nblocks
                    # g1 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
                    # g2 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
                    # g1.gsoft_L.data = L1.clone()
                    # g2.gsoft_L.data = L2.clone()
                    # g1.gsoft_R.data = R1.clone()
                    # g2.gsoft_R.data = R2.clone()
                    # merged_gs = blocked_geodesic_combination(g1, g2, t)
                    # #merged_gs = full_matrix_geodesic_combination(g1, g2, t)
                    # getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.gsoft_L.data)
                    # getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.gsoft_R.data)
                    #print("merging_type:", merging_type)
                    if merging_type == "blocked_right":
                        # if count == 0:
                        #     L_before = getattr(proc1, layer).ort_monarch.L.data
                        #     R_before = getattr(proc1, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_before_blocked_1_blocked_right.npy'), L_before.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_before_blocked_1_blocked_right.npy'), R_before.detach().cpu().numpy())
                        #     L_before = getattr(proc2, layer).ort_monarch.L.data
                        #     R_before = getattr(proc2, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_before_blocked_2_blocked_right.npy'), L_before.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_before_blocked_2_blocked_right.npy'), R_before.detach().cpu().numpy())
                            
                            
                        #print("getattr(proc1, layer).ort_monarch.L.data.shape:", getattr(proc1, layer).ort_monarch.L.data.shape)
                        #print("getattr(proc1, layer).ort_monarch.R.data.shape:", getattr(proc1, layer).ort_monarch.R.data.shape)
                        merged_gs = blocked_geodesic_combination(getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch, t)
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"
                        
                        
                        # Save matrix after blocked merging
                        #monarch_merged = getattr(proc1, layer).ort_monarch
                        #torch.save(monarch_merged.state_dict(), os.path.join(matrices_dir, f'{name}_{layer}_monarch_after_blocked.pt'))
                        
                        # Save matrices after blocked merging
                        # if count == 0:
                        #     L_merged = getattr(proc1, layer).ort_monarch.L.data
                        #     R_merged = getattr(proc1, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_after_blocked_blocked_right.npy'), L_merged.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_after_blocked_blocked_right.npy'), R_merged.detach().cpu().numpy())
                        #     count += 1
                    elif merging_type == "blocked_double_ortog":
                        # if count == 0:
                        #     L_before = getattr(proc1, layer).ort_monarch.L.data
                        #     R_before = getattr(proc1, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_before_blocked_1_blocked_double_ortog.npy'), L_before.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_before_blocked_1_blocked_double_ortog.npy'), R_before.detach().cpu().numpy())
                        #     L_before = getattr(proc2, layer).ort_monarch.L.data
                        #     R_before = getattr(proc2, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_before_blocked_2_blocked_double_ortog.npy'), L_before.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_before_blocked_2_blocked_double_ortog.npy'), R_before.detach().cpu().numpy())
                            
                        merged_gs = blocked_geodesic_combination(getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch, t)
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        # if count == 0:
                        #     L_merged = getattr(proc1, layer).ort_monarch.L.data
                        #     R_merged = getattr(proc1, layer).ort_monarch.R.data
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_L_after_blocked_double_ortog.npy'), L_merged.detach().cpu().numpy())
                        #     np.save(os.path.join(matrices_dir, f'{name}_{layer}_R_after_blocked_double_ortog.npy'), R_merged.detach().cpu().numpy())
                        #     count += 1
    
                    elif merging_type == "full":
                        # print("L.shape:", getattr(proc1, layer).ort_monarch.L.data.shape)
                        # print("R.shape:", getattr(proc1, layer).ort_monarch.R.data.shape)
                        merged_gs = full_matrix_geodesic_combination(getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch, t)
                        #merged_gs = blocked_geodesic_combination(getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch, t)
                        
                        new_ort_monarch = MonarchOrthogonal(n=merged_gs.n, nblocks=merged_gs.nblocks, orthogonal=True, method=self.config['moft_method'])
                        new_ort_monarch.L.data = merged_gs.L.data
                        new_ort_monarch.R.data = merged_gs.R.data
                        new_ort_monarch.R.data = new_ort_monarch.R.data.unsqueeze(0)
                        # print("new_ort_monarch.L.data.shape:", new_ort_monarch.L.data.shape)
                        # print("new_ort_monarch.R.data.shape:", new_ort_monarch.R.data.shape)
                        # new_ort_monarch.L.data.shape: torch.Size([1, 640, 640])
                        # new_ort_monarch.R.data.shape: torch.Size([640, 640])
                        # L.shape: torch.Size([32, 20, 20])
                        # R.shape: torch.Size([32, 20, 20])
                        
                        # new_ort_monarch.L.data.shape: torch.Size([1, 640, 640])
                        # new_ort_monarch.R.data.shape: torch.Size([1, 640, 640])
                        # L.shape: torch.Size([32, 20, 20])
                        # R.shape: torch.Size([32, 20, 20])
                        
                        getattr(proc1, layer).ort_monarch = new_ort_monarch
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"
                        
                        # Save matrix after full merging
                        #torch.save(new_ort_monarch.state_dict(), os.path.join(matrices_dir, f'{name}_{layer}_monarch_after_full.pt'))
                    elif merging_type == "riemannian":
                        #merged_gs = blocked_geodesic_combination(getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch, t)
                        merged_gs = riemannian_barycenter_approximation([getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch], ts=torch.tensor([t, 1-t]), als_steps=300, device=self.device)
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"
                        
                    elif merging_type == "fast_riemannian":
                        #алс наверное стоит запускать где-то с числом шагов порядка 100 для матриц размером поменьше (640 на 640 и число блоков 32) и примерно 750 для блоков большего размера (1280 и 32 блока)
                        print("getattr(proc1, layer).ort_monarch.n:", getattr(proc1, layer).ort_monarch.n)
                        if getattr(proc1, layer).ort_monarch.n < 641:
                            als_steps = 100
                        else:
                            als_steps = 750
                        merged_gs = fast_riemannian_barycenter_approximation([getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch], ts=torch.tensor([t, 1-t]), als_steps=als_steps, device=self.device, svd_frequency=8, initialize_in_heaviest_point=True)
                        merged_gs = merged_gs["barycenter_approximation"]
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"
                    
                    elif merging_type == "fast_riemannian_without_svd":
                        #алс наверное стоит запускать где-то с числом шагов порядка 100 для матриц размером поменьше (640 на 640 и число блоков 32) и примерно 750 для блоков большего размера (1280 и 32 блока)
                        print("getattr(proc1, layer).ort_monarch.n:", getattr(proc1, layer).ort_monarch.n)
                        if getattr(proc1, layer).ort_monarch.n < 641:
                            als_steps = 100
                        else:
                            als_steps = 750
                        merged_gs = fast_riemannian_barycenter_approximation_without_svd([getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch], ts=torch.tensor([t, 1-t]), als_steps=als_steps, device=self.device, svd_frequency=8, initialize_in_heaviest_point=True)
                        merged_gs = merged_gs["barycenter_approximation"]
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"

                    elif merging_type == "riemannian_2_0":
                        #алс наверное стоит запускать где-то с числом шагов порядка 100 для матриц размером поменьше (640 на 640 и число блоков 32) и примерно 750 для блоков большего размера (1280 и 32 блока)
                        print("getattr(proc1, layer).ort_monarch.n:", getattr(proc1, layer).ort_monarch.n)
                        als_steps = 150

                        merged_gs = riemannian_barycenter_approximation2_0([getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch], ts=torch.tensor([t, 1-t]), als_steps=als_steps, device=self.device, initialize_in_heaviest_point=True)
                        merged_gs = merged_gs["barycenter_approximation"]
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "already_orthogonal"
                    
                    elif merging_type == "merge_in_cayley_space":
                        merged_gs = merge_inside_cayley_space(
                                [getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch],
                                torch.Tensor([t, 1-t], device=getattr(proc1, layer).ort_monarch.L.data.device)
                        )
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        assert getattr(proc1, layer).ort_monarch.method == "cayley", "sanity check failed, method isn't cayley"
                    
                    elif merging_type == "merge_in_cayley_space_postprocess":
                        merged_gs = merge_inside_cayley_space(
                                [getattr(proc1, layer).ort_monarch, getattr(proc2, layer).ort_monarch],
                                torch.Tensor([t, 1-t], device=getattr(proc1, layer).ort_monarch.L.data.device)
                        )
                        rotation_angle = 0
                        alpha = 0
                        uniform_range = 0
                        merged_gs = postprocess_blocks(merged_gs, eta, rotation_angle, alpha, uniform_range)
                        getattr(proc1, layer).ort_monarch.L.data.copy_(merged_gs.L.data)
                        getattr(proc1, layer).ort_monarch.R.data.copy_(merged_gs.R.data)
                        getattr(proc1, layer).ort_monarch.method = "cayley" # I did it on purpose             
                    
                    else:
                        raise ValueError(f"Invalid merging type: {merging_type}")

 
        
    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model")
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
        print("setup_model")
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
        #print("moft_attn_procs", moft_attn_procs)
        self.unet.set_attn_processor(moft_attn_procs)
        self.moft_layers = AttnProcsLayers(self.unet.attn_processors)
        #self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_dog6.safetensors')))
        self.moft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_concept.safetensors')))
        
        #print("moft_attn_procs_style", moft_attn_procs_style)
        self.unet_style.set_attn_processor(moft_attn_procs_style)
        self.moft_layers_style = AttnProcsLayers(self.unet_style.attn_processors)
        #self.moft_layers_style.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_style3.safetensors')))
        self.moft_layers_style.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights_style.safetensors')))

        print("Concept L norm:", self.unet.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_moft.ort_monarch.L.data.norm())
        print("Style L norm:", self.unet_style.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_moft.ort_monarch.L.data.norm())

        # Merge to_q_moft weights (L and R) between self.unet and self.unet_style
        #from moft.mo.utils.gs_orthogonal import full_matrix_geodesic_combination, GSOrthogonal
        if self.args.t is not None:
            t = self.args.t
        else:
            print("WARNING: t is not provided, using midpoint merge")
            t = 0.5  # midpoint merge
        #merging_type = "blocked"
        #merging_type = "riemannian"
        #merging_type = "fast_riemannian"
        #merging_type = "fast_riemannian_without_svd"
        #merging_type = "full"
        #merging_type = "riemannian_2_0"
        # merging_type = "blocked_double_ortog"
        #merging_type = "blocked_right"
        #merging_type = "merge_in_cayley_space"
        merging_type = "merge_in_cayley_space_postprocess"
        print("eta:", self.args.eta)
        merging = True
        import torch
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU Name:", torch.cuda.get_device_name(0))
        else:
            print("NO GPU AVAILABLE!")
        if merging == True:
            print("Starting merging with merging_type:", merging_type, "and t:", t)
            print("merging to_q_moft...")
            self.layer_merge(t, layer = "to_q_moft", independent_merge=False, merging_type=merging_type, eta=self.args.eta)
            print("merging to_k_moft...")
            self.layer_merge(t, layer = "to_k_moft", independent_merge=False, merging_type=merging_type, eta=self.args.eta)
            print("merging to_v_moft...")
            self.layer_merge(t, layer = "to_v_moft", independent_merge=False, merging_type=merging_type, eta=self.args.eta)
            print("merging to_out_moft...")
            self.layer_merge(t, layer = "to_out_moft", independent_merge=False, merging_type=merging_type, eta=self.args.eta)
            print("merging done!")
        # Dog L norm: tensor(0.3732)
        # Style L norm: tensor(0.3842)
        # Merged L norm: tensor(0.3732)     0
        # Merged L norm: tensor(0.3676)     0.5
        # Merged L norm: tensor(0.3634)     1
        
        # Merged L norm: tensor(25.2982)     0
        # Merged L norm: tensor(25.2982)     0.5
        # Merged L norm: tensor(25.2982)     1
        # After merging
        # (same as above, but after merge)
        print("Merged L norm:", self.unet.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_moft.ort_monarch.L.data.norm())
        
        # for name, proc1 in self.unet.attn_processors.items():
        #     proc2 = self.unet_style.attn_processors[name]
        #     if hasattr(proc1, "to_q_moft") and hasattr(proc2, "to_q_moft"):
        #         independent_merge = False
        #         if independent_merge:
        #             # Merge L
        #             L1 = proc1.to_q_moft.ort_monarch.L.data
        #             L2 = proc2.to_q_moft.ort_monarch.L.data
        #             nblocks, n1, n2 = L1.shape   #[nblocks, block_size, block_size] - так как матрицы блочные
        #             n = n1 * nblocks
        #             g1 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
        #             g2 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
        #             g1.gsoft_L.data = L1.clone()
        #             g2.gsoft_L.data = L2.clone()
        #             g1.gsoft_R.data.zero_()
        #             g2.gsoft_R.data.zero_()
        #             merged_gs = blocked_geodesic_combination(g1, g2, t)
        #             proc1.to_q_moft.ort_monarch.L.data.copy_(merged_gs.gsoft_L.data)

        #             # Merge R
        #             R1 = proc1.to_q_moft.ort_monarch.R.data
        #             R2 = proc2.to_q_moft.ort_monarch.R.data
        #             g1.gsoft_R.data = R1.clone()
        #             g2.gsoft_R.data = R2.clone()
        #             g1.gsoft_L.data.zero_()
        #             g2.gsoft_L.data.zero_()
        #             merged_gs = blocked_geodesic_combination(g1, g2, t)
        #             proc1.to_q_moft.ort_monarch.R.data.copy_(merged_gs.gsoft_R.data)
                
        #         else:
        #             L1 = proc1.to_q_moft.ort_monarch.L.data
        #             L2 = proc2.to_q_moft.ort_monarch.L.data
        #             R1 = proc1.to_q_moft.ort_monarch.R.data
        #             R2 = proc2.to_q_moft.ort_monarch.R.data
                    
        #             nblocks, n1, n2 = L1.shape   #[nblocks, block_size, block_size] - так как матрицы блочные
        #             n = n1 * nblocks
        #             g1 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
        #             g2 = GSOrthogonal(n, nblocks, orthogonal=True, method="already_orthogonal")
        #             g1.gsoft_L.data = L1.clone()
        #             g2.gsoft_L.data = L2.clone()
        #             g1.gsoft_R.data = R1.clone()
        #             g2.gsoft_R.data = R2.clone()
        #             merged_gs = blocked_geodesic_combination(g1, g2, t)
        #             proc1.to_q_moft.ort_monarch.L.data.copy_(merged_gs.gsoft_L.data)
        #             proc1.to_q_moft.ort_monarch.R.data.copy_(merged_gs.gsoft_R.data)
        
        

@inferencers.add_to_registry('lora_merge')
class LoraMergeInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)
    
    def create_folder_name(self):
        print(f"create_folder_name: ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}")
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}_t{self.args.t}" 
    
    def setup_base_model(self):
        # Here we create base models
        print("setup_base_model")
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

        print("Concept Lora norm:", self.unet.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_lora.down.weight.norm())
        print("Style Lora norm:", self.unet_style.attn_processors['down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'].to_q_lora.down.weight.norm())

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
                    # For MOFT: A_proc1 = getattr(proc1, layer).ort_monarch.L.data
                    # For LoRA: A_proc1 = getattr(proc1, layer).down.weight
                    #           B_proc1 = getattr(proc1, layer).up.weight
                    if hasattr(getattr(proc1, layer), 'down'):
                        A_proc1 = getattr(proc1, layer).down.weight
                        B_proc1 = getattr(proc1, layer).up.weight
                        A_proc2 = getattr(proc2, layer).down.weight
                        B_proc2 = getattr(proc2, layer).up.weight
                        # print("A_proc1 shape:", A_proc1.shape)
                        # print("B_proc1 shape:", B_proc1.shape)
                        # print("A_proc2 shape:", A_proc2.shape)
                        # print("B_proc2 shape:", B_proc2.shape)
                        #print(dir(getattr(proc1, layer)))

                        fr_batch = [
                            (A_proc1.T, B_proc1),
                            (A_proc2.T, B_proc2),
                        ] 

                        conv_coef = torch.tensor([t, 1-t])

                        frb = FixedRankBatch(fr_batch, conv_coef)
                        approx = frb.riemannian_barycenter_approximation()
                        new_matrix = approx.to_dense()
                        #print("new_matrix shape:", new_matrix.shape)
                        #print("approx.V.shape:", approx.V.shape)
                        #print("approx.U.shape:", approx.U.shape)
                        #getattr(proc1, layer).ort_monarch.R.data.copy_(approx)
                        with torch.no_grad():
                            getattr(proc1, layer).down.weight.copy_(approx.U.T)
                            getattr(proc1, layer).up.weight.copy_(approx.V)
                        

                        #print(a.shape)
 
                else:
                    raise ValueError(f"Invalid merging type: {merging_type}")

 

