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

def noise_w_seed(seed):
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(torch_device)

    return noise

unet = UNet2DModel_H.from_pretrained("google/ddpm-ema-celebahq-256")
scheduler = DDIMScheduler.from_config("google/ddpm-ema-celebahq-256")
scheduler.set_timesteps(num_inference_steps=20)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(torch_device)


#############################################################

seeds = [144, 143] #[125, 133, 135, 138, 143, 144]
noises = []
h_features = {(seed, n):[] for seed in seeds for n in range(len(scheduler.timesteps))}
h_dists = []
h_paths = []
step = 0.2

for s, seed in enumerate(seeds):

    for w in np.arange(step, 1 + step, step):
        # image = slerp(w, noise_w_seed(seeds[n]), noise_w_seed(seeds[n-1]))
        image = torch.lerp(noise_w_seed(seeds[s]), noise_w_seed(seeds[s-1]), w)

        for n, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                residual = unet(image, t)[0]["sample"]
                h_feature = unet(image, t)[1] # (1, 896, 8, 8)
                h_features[seed, n].append(h_feature)
            prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]
            image = prev_image

for s, seed in enumerate(seeds):
    print(f"Currently n={s}, seed={seed}")
    
    for n in range(len(scheduler.timesteps)-1):
        
        dist = ((h_features[seed, n][0] - h_features[seed, n][-1]) ** 2).sum().sqrt()
        h_dists.append(dist.cpu())

        path = 0
        for w in range(int(1/step) - 1):
            path += ((h_features[seed, n][w] - h_features[seed, n][w+1]) ** 2).sum().sqrt()
        h_paths.append(path.cpu())

    print(h_dists)
    print(h_paths)

    plt.plot(h_dists, label = 'h_dists')
    plt.plot(h_paths, label = 'h_paths')

    plt.xlabel('Timesteps')
    plt.ylabel('L2 distance')

    plt.legend()
    plt.savefig(f"output/h_pathlength_exp")

    if s==0:
        break

print("Done!")