from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from torchvision.utils import save_image
import torch

model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a professional photograph of two people with one wearing a baseball cap."
num_images = 8
device = "cuda"

generator = torch.Generator(device=device)
latents = None
seeds = []
images = []
features = []

for i in range(num_images):
    # Get a new random seed, store it and use it as the generator state
    seed = generator.seed()
    seeds.append(seed)
    generator = generator.manual_seed(seed)
    
    image_latents = torch.randn(
        (1, pipe.unet.in_channels, 768 // 8, 768 // 8),
        generator = generator,
        device = device,
        dtype=torch.float16
    )

    latents = image_latents #torch.cat((image_latents, image_latents))

    image = pipe(
        prompt,
        # output_type = "latent",
        latents = latents,
        ).images[0]
    images.append(image)

    feature = pipe(
        prompt,
        output_type = "latent",
        latents = latents,
        ).images[0]
    features.append(feature)

    print(f"{i+1}th image generated.")

for i, img in enumerate(images):
    img.save(f"output/stable-diffusion-2/image_{i}.png")

for i, feature in enumerate(features):
    save_image(feature, f"output/stable-diffusion-2-latent/image_{i}.png")

print(seeds)
