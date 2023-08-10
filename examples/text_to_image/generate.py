from diffusers import StableDiffusionPipeline
import torch

model_name = "echarlaix/tiny-random-stable-diffusion-xl-refiner" #"stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A majestic lion jumping from a big stone at night"
# prompt = "old male human wizard wearing yellow and black robes photorealistic, very long straight white and grey hair, grey streaks, ecstatic, (60-year old Austrian male:1.1), sharp, (older body:1.1), stocky, realistic, real shadow 3d, (highest quality), (concept art, 4k), (wizard labratory in backgound:1.2), by Michelangelo and Alessandro Casagrande and Greg Rutkowski and Sally Mann and jeremy mann and sandra chevrier and maciej kuciara, inspired by (arnold schwarzenegger:1.001) and (Dolph Lundgren:1.001) and (Albert Einstien:1.001)"

num_images = 4
images = []
for i in range(num_images):
    with torch.no_grad():
        image = pipe(prompt=prompt).images[0]
        images.append(image)

for i, img in enumerate(images):
    img.save(f"./output/tiny/image_{i}.png")
