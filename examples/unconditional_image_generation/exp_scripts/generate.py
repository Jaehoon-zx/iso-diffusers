from diffusers import DDPMPipeline, StableDiffusionPipeline, DiffusionPipeline
import torch

model_name = "mrm8488/ddpm-ema-butterflies-128"
pipe = DiffusionPipeline.from_pretrained(model_name)
pipe.to("cuda")
print(pipe)
# prompt = "A majestic lion jumping from a big stone at night"

seeds = range(30)

images = []
for i, seed in enumerate(seeds):
    print(f"{i}th image generating.")
    with torch.no_grad():
        generator = torch.manual_seed(seed)
        image = pipe(
            num_inference_steps = 50,
            generator=generator
        ).images
        images.append(image)

images = [x for image in images for x in image]
for i, img in enumerate(images):
    img.save(f"./output/butterfly/image_{i}.png")
