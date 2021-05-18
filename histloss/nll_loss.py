import torch
from .base_hist_loss import BaseHistLoss
from .utils import norm_min_max_distributuions

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
        self.eps = eps

    def forward(self, positive, negative):
        pos_loss = -torch.log(positive + self.eps).mean()
        neg_loss = -torch.log(1 - negative + self.eps).mean()
        loss = pos_loss + neg_loss
        return loss
    