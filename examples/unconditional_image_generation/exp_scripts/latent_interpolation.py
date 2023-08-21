import torch

from diffusers import DiffusionPipeline, DDPMPipeline, UNet2DModel

# Stable Diffusion
# pipe = DiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     revision='fp16',
#     torch_dtype=torch.float16,
#     safety_checker=None,  # Very important for videos...lots of false positives while interpolating
#     custom_pipeline="interpolate_stable_diffusion",
# ).to('cuda')
# pipe.enable_attention_slicing()

# Load from my checkpoint/unet
model_path = "ddpm-ema-pokemon-64"
unet = UNet2DModel.from_pretrained("output/" + model_path + "/checkpoint-20000/unet")
pipe = DDPMPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16)
# pipe = DiffusionPipeline.from_pretrained(model_path).to("cuda")

frame_filepaths = pipe.walk(
    # prompts=['a boy', 'a house', 'a girl'],
    seeds=[42, 1337, 1234],
    num_interpolation_steps=16,
    output_dir='./pokemon-64',
    batch_size=4,
    height=256,
    width=256,
    # guidance_scale=8.5,
    num_inference_steps=50,
)
