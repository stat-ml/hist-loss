import torch
from torch import Tensor
from histloss.base_hist_loss import BaseHistLoss
from histloss.utils import norm_min_max_distributuions

class NLLLoss(torch.nn.Module):
    """
    Example of usage:
        >>> criterion = NLLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()
    """
    def __init__(self, eps=1e-15):
        super(NLLLoss, self).__init__()
        self._eps = eps

    def forward(self, positive: Tensor, negative: Tensor):
        pos_loss = -torch.log(positive + self._eps).mean()
        neg_loss = -torch.log(1 - negative + self._eps).mean()
        loss = pos_loss + neg_loss
        return loss
    