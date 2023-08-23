import torch
import numpy as np
import torch.autograd.functional as A


def stereographic_proj(x, device):
    # R x S^(n-1) -> R x R^(n-1)
    bs = x.shape[0]
    x = x.flatten(1)
    r = (x**2).sum(1).sqrt().unsqueeze(1)

    return torch.cat((r, (1/(r - x[:,-1:])) * x[:,:-1]), dim=-1)

def inv_stereographic_proj(t, shape ,device):
    #  R x R^(n-1) --> R x S^(n-1)
    t = t.flatten(1)
    r, t = t[:,:1], t[:,1:]
    t_norm_sq = (t ** 2).sum(1).unsqueeze(1)

    result = r * torch.cat((2 * t, t_norm_sq - 1), dim=-1) / (t_norm_sq + 1)

    return result.reshape(shape)

def riemmanian_metric(t, device):
    # Riemmanian metric in stereographic coordinates
    r, t = t[:,:1], t[:,1:]
    t_norm_sq = (t ** 2).sum()

    G_r = torch.ones((t.shape[0], 1), device=device)
    G_t = 4 * r ** 4/ (t_norm_sq + r ** 2) ** 2 * torch.ones_like(t)

    return torch.cat((G_r, G_t), dim=-1)

def isometry_loss(f, x, timesteps, device):
    bs = x.shape[0]
    t = stereographic_proj(x, device=device)
    G = riemmanian_metric(t, device=device)

    u = torch.randn_like(t, device=device)
    v = (1/G * u).reshape(x.shape)
    Ju = A.jvp(lambda t: f(inv_stereographic_proj(t, x.shape, device), timesteps)[1], t, 1/G * u, create_graph=True)[1] 
    JTJu = A.vjp(lambda t: f(inv_stereographic_proj(t, x.shape ,device), timesteps)[1], t, Ju, create_graph=True)[1]

    TrG = torch.sum(Ju.view(bs, -1) ** 2, dim=1).mean()
    TrG2 = torch.sum(JTJu.view(bs, -1) ** 2, dim=1).mean()

    isometry_loss = TrG2 / TrG ** 2

    return isometry_loss

def isometry_loss_test(f, x, timesteps, device):
    bs = x.shape[0]
    # t = stereographic_proj(x)
    # G = riemmanian_metric(t)

    u = torch.randn_like(x, device=device)
    Ju = A.jvp(lambda t: f(t, timesteps)[1], x, u, create_graph=True)[1] 
    JTJu = A.vjp(lambda t: f(t, timesteps)[1], x, Ju, create_graph=True)[1]

    TrG = torch.sum(Ju.view(bs, -1) ** 2, dim=1).mean()
    TrG2 = torch.sum(JTJu.view(bs, -1) ** 2, dim=1).mean()

    isometry_loss = TrG2 / TrG ** 2

    return isometry_loss