import torch
from diffusers import DiffusionPipeline, DDPMPipeline, UNet2DConditionModel

# Load from my checkpoint/unet
model_path = "output/sd2-ffhq512-3"
model_name = "stabilityai/stable-diffusion-2"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-69000/unet")
pipe = DiffusionPipeline.from_pretrained(
    model_name, 
    unet=unet, 
    #torch_dtype=torch.float16,
    safety_checker=None,
    custom_pipeline="interpolate_stable_diffusion",
).to("cuda")

frame_filepaths = pipe.walk(
    prompts=["a photography of a woman not smiling",
             "a photography of a woman smiling"
    ],
    seeds=[1, 2],
    num_interpolation_steps=64,
    output_dir='output/ffhq-interpolation',
    batch_size=4,
    height=512,
    width=512,
    guidance_scale=8.5,
    num_inference_steps=50,
)
