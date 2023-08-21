from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A majestic lion jumping from a big stone at night"

num_images = 4
images = []
for i in range(num_images):
    image = pipe(prompt=prompt).images[0]
    images.append(image)

for i, img in enumerate(images):
    img.save(f"./output/sd-xl-base/image_{i}.png")