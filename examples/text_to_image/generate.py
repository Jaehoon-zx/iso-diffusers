from diffusers import DDPMPipeline, StableDiffusionPipeline, DiffusionPipeline
import torch

model_name = "CompVis/ldm-celebahq-256" #"stabilityai/stable-diffusion-xl-base-1.0"

# pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained(model_name)
pipe.to("cuda")

print(pipe)

prompt = "A majestic lion jumping from a big stone at night"

num_images = 4
batch_size = 2
images = []
for i in range(num_images // batch_size):
    with torch.no_grad():
        image = pipe(
            batch_size = batch_size,
            num_inference_steps = 50,
        ).images
        images.append(image)

images = [x for image in images for x in image]
for i, img in enumerate(images):
    img.save(f"./output2/ldm-256/image_{i}.png")
