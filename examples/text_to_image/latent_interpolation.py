import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, UNet2DConditionModel
from stable_diffusion_videos import StableDiffusionWalkPipeline

# Pre-trained model
pipe = DiffusionPipeline.from_pretrained(
    "google/ddpm-ema-celebahq-256",
    # revision='fp16',
    # torch_dtype=torch.float16,
    safety_checker=None,  # Very important for videos...lots of false positives while interpolating
    custom_pipeline="interpolate_stable_diffusion",
).to('cuda')
pipe.enable_attention_slicing()

# pipe = StableDiffusionWalkPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2",
#     torch_dtype=torch.float16,
#     safety_checker=None
# ).to("cuda")

# Load from my checkpoint/unet
# model_path = "output/sd2-ffhq512-43"
# model_name = "lambdalabs/miniSD-diffusers"
# unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-320000/unet")

# pipe = DiffusionPipeline.from_pretrained(
#     model_name, 
#     unet=unet, 
#     #torch_dtype=torch.float16,
#     safety_checker=None,
#     custom_pipeline="interpolate_stable_diffusion",
# ).to("cuda")

frame_filepaths = pipe.walk(
    # prompts=["a professional photography of an astronaut riding a horse",
    #          "a professional photography of a cowboy riding a horse",
    # ],
    prompts=["",
            "",
    ],
    seeds=[42, 1337],
    num_interpolation_steps=16,
    output_dir='output2/ddpm-256-interp',
    batch_size=2,
    height=256,
    width=256,
    # guidance_scale=8.5,
    num_inference_steps=20,
)
