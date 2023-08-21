from diffusers import UNet2DModel_H, DDIMScheduler, VQModel
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


seeds = [144, 143] #[125, 133, 135, 138, 143, 144]
noises = []

# load all models
unet = UNet2DModel_H.from_pretrained("google/ddpm-ema-celebahq-256")
# vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_config("google/ddpm-ema-celebahq-256")

# set to cuda
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(torch_device)
# vqvae.to(torch_device)

h_features = {seed:[] for seed in seeds}

for seed in seeds:
    # generate gaussian noise to be decoded
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(torch_device)
    noises.append(noise)

    # set inference steps for DDIM
    scheduler.set_timesteps(num_inference_steps=20)
    image = noise
    fig, axes = plt.subplots(nrows=2 ,ncols=5, figsize=(6, 5))

    for t in scheduler.timesteps:
        # predict noise residual of previous image
        with torch.no_grad():
            residual = unet(image, t)[0]["sample"]
            h_feature = unet(image, t)[1] # (1, 896, 8, 8)
            h_features[seed].append(h_feature)
            # fig, axes = plt.subplots(nrows=2 ,ncols=5, figsize=(6, 5))

            # for channel_idx, ax in enumerate(axes.flatten()):
            #     ax.imshow(h_feature[0, channel_idx].cpu(), cmap='viridis', interpolation='nearest')
            #     ax.set_title(f'{channel_idx + 1}')
            #     ax.axis('off')

            # plt.savefig(f"output/h_feature_{t}.png")
            # save_image(h_feature[0, 0], f"output/0th_h_feature_{t}.png")

        #     axes = axes.flatten()
        #     axes[t//100].imshow(h_feature[0, 1].cpu(), cmap='viridis', interpolation='nearest')
        #     axes[t//100].set_title(f't={t}')
        #     axes[t//100].axis('off')

        # plt.savefig(f"output/h_feature_over_t.png")

        prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
        image = prev_image

    # decode image with vae
    # with torch.no_grad():
    #     image = vqvae.decode(image)

    # process image
    # image_processed = image.cpu().permute(0, 2, 3, 1)
    # image_processed = (image_processed + 1.0) * 127.5
    # image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
    # image_pil = PIL.Image.fromarray(image_processed[0])

    # image_pil.save(f"output/ddpm4/image_{seed}.png")


for n, seed in enumerate(seeds):
    print(f"Currently n={n}, seed={seed}")
    # generate gaussian noise to be decoded
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(torch_device)

    # set inference steps for DDIM
    # scheduler.set_timesteps(num_inference_steps=10)
    # x/10 for x in range(0,11,1)

    print(f"n={n}")

    step = 0.1
    for w in np.arange(0, 1 + step, step):
        image = torch.lerp(noises[n], noises[n-1], w)
        # image = slerp(w, noises[n], noises[n-1])
        image_no = image
        image_slerp = image
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                residual = unet(image, t, h_feature_in= torch.lerp(h_features[seeds[n]][i], h_features[seeds[n-1]][i], w) )[0]["sample"]
                residual_slerp = unet(image_slerp, t, h_feature_in= slerp(w, h_features[seeds[n]][i], h_features[seeds[n-1]][i]) )[0]["sample"]
                residual_no = unet(image_no, t)[0]["sample"] 

            image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
            image_slerp = scheduler.step(residual_slerp, t, image_slerp, eta=0.0)["prev_sample"]
            image_no = scheduler.step(residual_no, t, image_no, eta=0.0)["prev_sample"]

        # decode image with vae
        # with torch.no_grad():
        #     image = vqvae.decode(image)
        path = 'output/ddpm7'

        image_processed = image.cpu().permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
        image_pil = PIL.Image.fromarray(image_processed[0])
        image_pil.save(path + f"/noise_exp_{seeds[n]}->{seeds[n-1]}_{round(w,2)}.png")

        image_processed = image_slerp.cpu().permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
        image_pil = PIL.Image.fromarray(image_processed[0])
        image_pil.save(path + f"/noise_exp_slerp_{seeds[n]}->{seeds[n-1]}_{round(w,2)}.png")

        image_processed = image_no.cpu().permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
        image_pil = PIL.Image.fromarray(image_processed[0])
        image_pil.save(path + f"/noise_exp_no_{seeds[n]}->{seeds[n-1]}_{round(w,2):.2f}.png")
    
    if n==0:
        break

print("Done!")