import argparse
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
#from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer



#########################3

base_dir = '/home/ubuntu/ImgGen/lab5'
save_path = f'{base_dir}/inputs_textual_inversion'

concept_name = "Mikhail_Dzigen_Menshikov"

#`initializer_token` is a word that can summarise what your 
#new concept is, to be used as a starting point
initializer_token = "man" #@param {type:"string"}

# `what_to_teach`: what is it that you are teaching? `object` enables you to teach the model a new object to be used,
# `style` allows you to teach the model a new style one can use.
what_to_teach = "object" #@param ["object", "style"]

# `placeholder_token` is the token you are going to use to represent your new concept (so when you prompt the model, 
# you will say "A `<my-placeholder-token>` in an amusement park"). We use angle brackets to differentiate a token from other words/tokens, to avoid collision.
placeholder_token = f'<{concept_name}>'

hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": 2000,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "output_dir": f'{base_dir}/{concept_name}-concept',
}

#########################3

pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],  
    # "downloaded_embedding",
    #torch_dtype=torch.float16,
).to("cuda")


save_prompt_images = '1'
prompt = f"a good photo of {placeholder_token}, high quality" #@param {type:"string"}
num_samples = 1 #@param {type:"number"}
num_rows = 1 #@param {type:"number"}

print("Generating images...")

# prevent safety checking
def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy
 
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=500, guidance_scale=7.5).images
    
    for i in range(images):
        images[i].save(f"{base_dir}/prompt_dir{save_prompt_images}/{i}.jpg") 

print("Generating images...")

#####

