import torch
import copy
import numpy as np
# import lpips
import torch.distributed as dist
import pickle
import dnnlib
import numpy as np
import time
import os
# os.chdir("/data/projects/jaehoon/iso-diffusers/examples/text_to_image")
from scipy import linalg
from compute_fid_statistics import get_activations
from dnnlib.util import open_url
from torchvision.transforms import ToTensor, ToPILImage, Compose
from PIL import Image
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size

def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)
    return x

def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir='.stylegan2-cache')
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
        t = t.cpu().numpy()

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    print("s1 dot s2", sigma1.dot(sigma2))
    print("covmean", covmean)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(n_samples, n_gpus, sampling_shape, sampler, gen, stats_path, device, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    tf_toTensor = ToTensor()
    transform = Compose([ToTensor()])
    print(sampling_shape)

    def generator(num_samples):
        num_sampling_rounds = int(np.ceil(num_samples / sampling_shape[0]))
        with torch.autocast("cuda"):
            for _ in tqdm(range(num_sampling_rounds)):
                x = sampler(batch_size= sampling_shape[0], num_inference_steps=20, generator=gen, output_type='np').images
                x = torch.tensor(x * 255, device=device).to(torch.uint8) # Range Debugging
                x = x.permute(0, 3, 2, 1)
                yield x

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(device)
        inception_model.eval()

    act = get_activations(generator(num_samples_per_gpu), inception_model,
                          sampling_shape[0], device=device, max_samples=n_samples, dl_include_step=False)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()

    print(m.shape, s.shape)
    # average_tensor(m)
    # average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    print(all_pool_mean, all_pool_sigma)
    print()

    stats = np.load(stats_path)
    data_pools_mean = stats['mu']
    data_pools_sigma = stats['sigma']

    print(data_pools_mean, data_pools_sigma)

    fid = calculate_frechet_distance(data_pools_mean,
                data_pools_sigma, all_pool_mean, all_pool_sigma)
    return fid

def compute_ppl(n_samples, n_gpus, sampling_shape, sampler, gen, device, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    epsilon = 1e-4
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape):
        with torch.autocast("cuda"):
            with torch.no_grad():
                z0 = torch.randn(sampling_shape, device=device, dtype=sampler.unet.dtype)
                z1 = torch.randn(sampling_shape, device=device, dtype=sampler.unet.dtype)
                # y0 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                # y1 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                            
                t = torch.rand(sampling_shape[0], device=device, dtype=sampler.unet.dtype)
                t = add_dimensions(t, 3)

                zt0 = slerp(t, z0, z1)
                zt1 = slerp(t + epsilon, z0, z1)
                # yt0 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t).squeeze(1)
                # yt1 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t + epsilon).squeeze(1)

                x0 = sampler(latents=zt0, num_inference_steps=20, generator=gen, output_type='pt').images
                x1 = sampler(latents=zt1, num_inference_steps=20, generator=gen, output_type='pt').images
                x0 = torch.tensor(x0.transpose(0, 3, 1, 2), device=device)
                x1 = torch.tensor(x1.transpose(0, 3, 1, 2), device=device)
                x0 = (x0 * 2 - 1).clip(-1, 1)
                x1 = (x1 * 2 - 1).clip(-1, 1)
                imgs = (x0, x1)

        return imgs

    def calculate_lpips(x0, x1):
        # loss_fn_alex = lpips.LPIPS(net = 'alex', verbose = False).to(device)
        # dist = loss_fn_alex(x0, x1).square().sum((1,2,3)) / epsilon ** 2
        # print(loss_fn_alex(x0, x1).shape)

        # epsilon = 1e-6
        # distance_measure = load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/vgg16_zhang_perceptual.pkl')
        # dist = distance_measure.get_output_for(x0, x1) * (1 / epsilon**2)

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device) # input detail is described in "torchmetric docs"
        dist = lpips(x0, x1) / epsilon ** 2

        return dist

    dist_list = []
    for i in tqdm(range(0, n_samples, sampling_shape[0])):
        x0, x1 = generator(sampling_shape)
        dist = calculate_lpips(x0, x1)
        dist_list.append(dist.detach().cpu())

    # Compute PPL.
    dist_list = np.array(dist_list)
    lo = np.percentile(dist_list, 1, interpolation='lower')
    hi = np.percentile(dist_list, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist_list >= lo, dist_list <= hi), dist_list).mean()

    return float(ppl)


def compute_distortion_per_timesteps(n_samples, n_gpus, sampling_shape, sampler, gen, device, text=None, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    epsilon = 1e-3
    tf_toTensor = ToTensor()
    if type(text) is list:
        text = np.asarray(text)

    def generator(sampling_shape, interrupt_step):
        with torch.autocast("cuda"):
            with torch.no_grad():
                z0 = torch.randn(sampling_shape, device=device, dtype=sampler.text_encoder.dtype)
                z1 = torch.randn(sampling_shape, device=device, dtype=sampler.text_encoder.dtype)
                y0 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                y1 = sampler._encode_prompt(list(text[torch.randint(0,len(text),[sampling_shape[0]])]), device, 1, True)[:sampling_shape[0]]
                            
                t = torch.rand(sampling_shape[0], device=device, dtype=sampler.text_encoder.dtype)
                t = add_dimensions(t, 3)

                zt0 = slerp(t, z0, z1)
                zt1 = slerp(t + epsilon, z0, z1)
                yt0 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t).squeeze(1)
                yt1 = torch.lerp(y0.unsqueeze(1), y1.unsqueeze(1), t).squeeze(1)

                x0 = sampler(latents=zt0, prompt_embeds=yt0, num_inference_steps=50, generator=gen, output_type='latent', interrupt_step=interrupt_step).images
                x1 = sampler(latents=zt1, prompt_embeds=yt1, num_inference_steps=50, generator=gen, output_type='latent', interrupt_step=interrupt_step).images
                x0 = (x0 * 2 - 1).clip(-1., 1.)
                x1 = (x1 * 2 - 1).clip(-1., 1.)
                imgs = (x0, x1)

        return imgs

    def calculate_l2(x0, x1):
        dist = torch.sum((x0 - x1) ** 2, dim=(1,2,3))
        return dist

    dist_list_per_timesteps = []
    for j in range(1,51):
        print(f"Calculating {j}th timestep's distortion.")
        dist_list = []
        for i in range(0, n_samples, sampling_shape[0]):
            x0, x1 = generator(sampling_shape, interrupt_step=j)
            dist = calculate_l2(x0, x1)
            dist_list.append(dist.detach().cpu())

        dist_list = torch.cat(dist_list)[:n_samples].cpu().detach().numpy()
        print("dist_list=", dist_list)
        lo = np.percentile(dist_list, 10, interpolation='lower')
        hi = np.percentile(dist_list, 90, interpolation='higher')
        ppl = np.extract(np.logical_and(dist_list >= lo, dist_list <= hi), dist_list).mean()
        dist_list_per_timesteps.append(float(ppl))

    return dist_list_per_timesteps

#################################################################

def get_random_local_basis(model, random_state, noise = None, noise_dim = 512):
    '''
    noise_dim = 512 for StyleGAN, 128 for BigGAN
    
    ex)
    random_state = np.random.RandomState(seed)
    noise, z, z_local_basis, z_sv = get_random_local_basis(model, random_state)
    '''
    n_samples = 1
    if noise is not None:
        assert(list(noise.shape) == [n_samples, noise_dim])
        noise = noise.detach().float().to(model.device)
    else:
        noise = torch.from_numpy(
                random_state.standard_normal(noise_dim * n_samples)
                .reshape(n_samples, noise_dim)).float().to(model.device) #[N, noise_dim]
    noise.requires_grad = True
    
    if isinstance(model, StyleGAN2):
        mapping_network = model.model.style
    elif isinstance(model, StyleGAN):
        mapping_network = model.model._modules['g_mapping'].forward 
    elif isinstance(model, BigGAN):
        mapping_network = model.partial_forward_explicit
    else:
        raise NotImplemented   
    z = mapping_network(noise)

    ''' Compute Jacobian by batch '''
    noise_dim, z_dim = noise.shape[1], z.shape[1]
    noise_pad = noise.repeat(z_dim, 1).requires_grad_(True)
    z_pad = mapping_network(noise_pad)

    grad_output = torch.eye(z_dim).cuda()
    jacobian = torch.autograd.grad(z_pad, noise_pad, grad_outputs=grad_output, retain_graph=True)[0].cpu()
    
    ''' Get local basis'''
    # jacobian \approx torch.mm(torch.mm(z_basis, torch.diag(s)), noise_basis.t())
    z_basis, s, noise_basis = torch.svd(jacobian)
    return noise, z.detach(), z_basis.detach(), s.detach(), noise_basis.detach()


def compute_geodesic_metric(local_basis_1, local_basis_2, subspace_dim):
    subspace_1 = np.array(local_basis_1[:, :subspace_dim])
    subspace_2 = np.array(local_basis_2[:, :subspace_dim])
    
    u, s, v = np.linalg.svd(np.matmul(subspace_1.transpose(), subspace_2))
    s[s > 1] = 1
    s = np.arccos(s)
    return np.linalg.norm(s)