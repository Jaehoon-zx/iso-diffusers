import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, DiffusionPipeline, DDPMPipeline, UNet2DConditionModel
from stable_diffusion_videos import StableDiffusionWalkPipeline

# Pre-trained model
# pipe = DiffusionPipeline.from_pretrained(
#     # "stabilityai/stable-diffusion-2",
#     "CompVis/stable-diffusion-v1-4",
#     revision='fp16',
#     torch_dtype=torch.float16,
#     safety_checker=None,  # Very important for videos...lots of false positives while interpolating
#     custom_pipeline="interpolate_stable_diffusion",
# ).to('cuda')
# pipe.enable_attention_slicing()

# pipe = StableDiffusionWalkPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2",
#     torch_dtype=torch.float16,
#     safety_checker=None
# ).to("cuda")

# Load from my checkpoint/unet
model_path = "output/sd2-ffhq512-32"
model_name = "stabilityai/stable-diffusion-2"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-20000/unet")

pipe = DiffusionPipeline.from_pretrained(
    model_name, 
    unet=unet, 
    #torch_dtype=torch.float16,
    safety_checker=None,
    custom_pipeline="interpolate_stable_diffusion",
).to("cuda")

frame_filepaths = pipe.walk(
    # prompts=["a professional photography of an astronaut riding a horse",
    #          "a professional photography of a cowboy riding a horse",
    # ],
    prompts=["",
             "a photography of a little girl wearing pink dress smiling",
            "a photography of a woman smiling with her hands on her chin",
            "a photography of a man with glasses and a microphone in his hand",  
    ],
    seeds=[42, 1337, 648, 123],
    num_interpolation_steps=16,
    output_dir='output/sd2-interpolation',
    batch_size=8,
    height=256,
    width=256,
    guidance_scale=8.5,
    num_inference_steps=50,
)
