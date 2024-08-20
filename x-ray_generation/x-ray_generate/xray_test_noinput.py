import os
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoPipelineForImage2Image,
    StableDiffusionInstructPix2PixPipeline,
)
import torch
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer

accelerator = Accelerator(
    gradient_accumulation_steps= 1
)
generation_modol='Spinal-AI2024-advanced'
pretrained_model_name_or_path = os.path.join(generation_modol,"lora_weight_e30_s15000")

image_dir = "original-X-ray"
image_path = os.listdir(image_dir)
image_len = len(image_path)
save_dir = 'generation-X-ray'

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

if torch.cuda.is_available():
    print("true")
tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
        )

'''
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
        )
'''
pipeline.to("cuda")
i = 1
for ind in image_path:
    image_in = Image.open(os.path.join(image_dir, ind))
    image_out = pipeline(    
                            prompt = "X-ray"
                        ).images[0]    
    image_out.save(os.path.join(save_dir, ind)) 
    print("image{}generate successfully ,total iamges{}".format(ind, i))
    i+=1
    
    