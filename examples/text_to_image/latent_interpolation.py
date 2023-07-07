import torch

from diffusers import DiffusionPipeline, DDPMPipeline, UNet2DConditionModel

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
model_path = "sd-pokemon-model-8"
model_name = "stabilityai/stable-diffusion-2"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-2000/unet")
pipe = DiffusionPipeline.from_pretrained(
    model_name, 
    unet=unet, 
    #torch_dtype=torch.float16,
    safety_checker=None,
    custom_pipeline="interpolate_stable_diffusion",
).to("cuda")
# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)


frame_filepaths = pipe.walk(
    prompts=["a drawing of a blue and yellow pokemon", "a drawing of a black and white pokemon"],
    seeds=[42, 1337],
    num_interpolation_steps=100,
    output_dir='./pokemon-interpolation',
    batch_size=4,
    height=768,
    width=768,
    guidance_scale=8.5,
    num_inference_steps=50,
)
