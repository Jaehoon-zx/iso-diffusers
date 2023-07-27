import torch
import copy
import numpy as np
import lpips
import torch.distributed as dist
import pickle

from scipy import linalg
from compute_fid_statistics import get_activations
from dnnlib.util import open_url
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size

def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)
    return x

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

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(n_samples, n_gpus, sampling_shape, sampler, gen, stats_path, device, text, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    # torchvision.transforms.ToTensor
    tf_toTensor = ToTensor()
    def generator(num_samples):
        num_sampling_rounds = int(np.ceil(num_samples / sampling_shape[0]))
        with torch.autocast("cuda"):
            for _ in range(num_sampling_rounds):
                x = tf_toTensor(sampler(text[torch.randint(0,len(text),[1])], num_inference_steps=20, generator=gen).images[0])
                x = (x / 2. + .5).clip(0., 1.)
                x = (x * 255.).to(torch.uint8)
                yield x

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        inception_model = pickle.load(f).to(device)
        inception_model.eval()

    act = get_activations(generator(num_samples_per_gpu), inception_model,
                          sampling_shape[0], device=device, max_samples=n_samples, include_step=False)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()
    # average_tensor(m)
    # average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    stats = np.load(stats_path)
    data_pools_mean = stats['mu']
    data_pools_sigma = stats['sigma']
    fid = calculate_frechet_distance(data_pools_mean,
                data_pools_sigma, all_pool_mean, all_pool_sigma)
    return fid

######################################################################

def compute_ppl(n_samples, n_gpus, sampling_shape, sampler, vgg_model, device, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))
    epsilon = 1e-2

    def generator(num_samples):
        num_sampling_rounds = int(
            np.ceil(num_samples / sampling_shape[0]))
        for n in range(num_sampling_rounds):
            z0 = torch.randn(sampling_shape, device=device)
            z1 = torch.randn(sampling_shape, device=device)
            t = torch.randn(sampling_shape[0], device=device)
            t = add_dimensions(t, 3)

            zt0 = z0.lerp(z1, t)
            zt1 = z0.lerp(z1, t + epsilon)

            if n_classes is not None:
                y = torch.randint(n_classes, size=(
                    sampling_shape[0],), dtype=torch.int32, device=device)
                x0, _ = sampler(zt0, y=y)
                x1, _ = sampler(zt1, y=y)

            else:
                x0, _ = sampler(zt0)
                x1, _ = sampler(zt1)

            imgs = (x0, x1)

            for x in imgs:
                x = (x / 2. + .5).clip(0., 1.)
                x = (x * 255.).to(torch.uint8)
            
            return imgs

    def calculate_lpips(x0, x1):
        # vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
        # vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=opts.num_gpus, rank=opts.rank, verbose=opts.progress.verbose)

        # lpips_t0 = vgg16(x0, resize_images=False, return_lpips=True)
        # lpips_t0 = vgg16(x1, resize_images=False, return_lpips=True)

        loss_fn_alex = lpips.LPIPS(net = 'alex', verbose = False)

        dist = loss_fn_alex(x0.to('cpu'), x1.to('cpu')).square().sum((1,2,3)) / epsilon ** 2
        return dist

    dist_list = []
    for i in range(0, n_samples, sampling_shape[0]):
        x0, x1 = generator(num_samples_per_gpu)
        print(x0,x1)
        dist = calculate_lpips(x0, x1)
        print(dist)
        dist_list.append(dist)

    # Compute PPL.
    dist_list = torch.cat(dist_list)[:n_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()

    return float(ppl)