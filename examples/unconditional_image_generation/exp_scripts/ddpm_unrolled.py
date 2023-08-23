from diffusers import UNet2DModel_H, DDPMScheduler, DDIMScheduler, VQModel
from torchvision.utils import save_image
import torch
import PIL.Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
        t = t

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

seeds = range(30)
noises = []
h_features = {seed:[] for seed in seeds}
interp_cond = False
image_save = True
path = 'output/cats'

# load all models
# unet = UNet2DModel_H.from_pretrained("anton-l/ddpm-ema-flowers-64", subfolder='unet')
# vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
unet = UNet2DModel_H.from_pretrained("samwit/ddpm-afhq-cats-128", subfolder='unet')
scheduler = DDIMScheduler.from_config("samwit/ddpm-afhq-cats-128", subfolder='scheduler')
scheduler.set_timesteps(num_inference_steps=20)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(torch_device)

for seed in seeds:
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(torch_device)
    noises.append(noise)

    image = noise
    fig, axes = plt.subplots(nrows=2 ,ncols=5, figsize=(6, 5))

    for t in scheduler.timesteps:
        with torch.no_grad():
            residual = unet(image, t)[0]["sample"]
            h_feature = unet(image, t)[1]
            h_features[seed].append(h_feature)

        prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
        image = prev_image

    if image_save:
        image_processed = image.cpu().permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
        image_pil = PIL.Image.fromarray(image_processed[0])

        image_pil.save(path + f"/image_{seed}.png")

if interp_cond:
    for n, seed in enumerate(seeds):
        print(f"Currently n={n}, seed={seed}")
        # generate gaussian noise to be decoded
        generator = torch.manual_seed(seed)
        noise = torch.randn(
            (1, unet.in_channels, unet.sample_size, unet.sample_size),
            generator=generator,
        ).to(torch_device)

        step = 0.1
        for w in np.arange(0, 1 + step, step):
            # image = torch.lerp(noises[n], noises[n-1], w)
            image = slerp(w, noises[n], noises[n-1])
            image_no = image
            image_slerp = image
            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    residual = unet(image, t, h_feature_in= torch.lerp(h_features[seeds[n]][i], h_features[seeds[n-1]][i], w) )[0]["sample"]
                    # residual_slerp = unet(image_slerp, t, h_feature_in= slerp(w, h_features[seeds[n]][i], h_features[seeds[n-1]][i]) )[0]["sample"]
                    residual_no = unet(image_no, t)[0]["sample"] 

                image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
                # image_slerp = scheduler.step(residual_slerp, t, image_slerp, eta=0.0)["prev_sample"]
                image_no = scheduler.step(residual_no, t, image_no, eta=0.0)["prev_sample"]

            image_processed = image.cpu().permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
            image_pil = PIL.Image.fromarray(image_processed[0])
            image_pil.save(path + f"/{seeds[n]}->{seeds[n-1]}_{round(w,2)}.png")

            # image_processed = image_slerp.cpu().permute(0, 2, 3, 1)
            # image_processed = (image_processed + 1.0) * 127.5
            # image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
            # image_pil = PIL.Image.fromarray(image_processed[0])
            # image_pil.save(path + f"/noise_exp_slerp_{seeds[n]}->{seeds[n-1]}_{round(w,2)}.png")

            image_processed = image_no.cpu().permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
            image_pil = PIL.Image.fromarray(image_processed[0])
            image_pil.save(path + f"/no_{seeds[n]}->{seeds[n-1]}_{round(w,2):.2f}.png")
        
        if n==len(seeds)-2:
            break

print("Done!")