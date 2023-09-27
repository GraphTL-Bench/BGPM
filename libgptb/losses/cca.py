import torch
import numpy as np
import torch.nn.functional as F

from libgptb.losses.abstract_losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCESP(Loss):
    """
    InfoNCE loss for single positive.
    """
    def __init__(self, tau):
        super(InfoNCESP, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        sim = f(_similarity(anchor, sample))  # anchor x sample
        assert sim.size() == pos_mask.size()  # sanity check

        neg_mask = 1 - pos_mask
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class CCALoss():
    def __init__(self, lambd):
        super(CCALoss, self).__init__()
        self.lambd = lambd

    def compute(self, z1, z2) -> torch.FloatTensor:
        c = torch.mm(z1.T, z2) / z1.size(0)
        c1 = torch.mm(z1.T, z1) / z1.size(0)
        c2 = torch.mm(z2.T, z2) / z1.size(0)

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.size(0)), dtype=torch.float32).to(z1.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)

        return loss