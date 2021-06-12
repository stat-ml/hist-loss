import torch
from torch import nn, Tensor

class BinomialDevianceLoss(nn.Module):
    """
    Args:
        C (int, optional): asymmetric negative cost. Default: `10`
        alpha (float, optional): hyper-parameter. Default: `2`
        beta (float, optional): hyper-parameter. Default: `0.5`

    Shape:
        - positive: set of positive points, (N, *)
        - negative: set of negative points, (M, *)
        - loss: scalar

    Example of usage:
        >>> criterion = BinomialDevianceLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()

    Reference:
        Deep Metric Learning for Practical Person Re-Identification
        https://arxiv.org/pdf/1407.4979.pdf
    """
    def __init__(self, C: float = 10, alpha: float = 2, beta: float = 0.5):
        super(BinomialDevianceLoss, self).__init__()
        self.C = C
        self.alpha = alpha
        self.beta = beta

    def forward(self, positive: Tensor, negative: Tensor):
        loss_pos = (torch.log(torch.exp( -self.alpha * (positive - self.beta)) + 1)).mean()
        loss_neg = (torch.log(torch.exp(self.C * self.alpha * (negative - self.beta)) + 1)).mean()
        loss = loss_neg + loss_pos
        return loss
