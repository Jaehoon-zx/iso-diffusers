from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "drawing of a blue and yellow pokemon"
images = []
for i in range(8):
    image = pipe(prompt).images[0]
    images.append(image)

for i, img in enumerate(images):
    img.save(f"output/stable-diffusion-2/image_{i}.png")