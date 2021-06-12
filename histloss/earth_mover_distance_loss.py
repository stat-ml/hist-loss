import torch
from torch import Tensor
from histloss.base_hist_loss import BaseHistLoss
from histloss.utils import norm_min_max_distributuions

class EarthMoverDistanceLoss(BaseHistLoss):
    """
    EarthMoverDistanceLoss

    Args:
        bins (int, optional): .Default: `10`
        min_val (float, optional): Default: `-1`
        max_val (float, optional): Default: `1`
        alpha (float, optional): parameter for regularization. Default: `0`

    Shape:
        - positive: set of positive points, (N, *)
        - negative: set of negative points, (M, *)
        - loss: scalar

    Examples::
        >>> criterion = EarthMoverDistanceLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()
    """
    def __init__(self, bins: int = 128, alpha: float = 0,
                 method: str = 'asim', cut_negative: bool = True):
        super(EarthMoverDistanceLoss, self).__init__(bins=bins, alpha=alpha)
        if method not in {'sim', 'asim'}:
            raise NotImplementedError(f'Undefined method of EMD Loss: {method}')
        self.method = method
        self.cut_negative = cut_negative

    def forward(self, positive: Tensor, negative: Tensor):
        self.t = self.t.to(device=positive.device)
        positive, negative = norm_min_max_distributuions(positive, negative)
        
        if self.cut_negative:
            negative = negative[negative > (self._max_val - self._min_val) / 2]
        pos_hist = self.compute_histogram(positive) # h_pos
        neg_hist = self.compute_histogram(negative) # h_neg
        
        if self.method == 'sim':
            emd_loss = - (torch.abs(torch.cumsum(neg_hist - pos_hist, 0))).sum()
        elif self.method == 'asim':
            emd_loss = (torch.cumsum(pos_hist - neg_hist, 0)).sum()

        std_loss = self.std_loss(positive, negative)

        loss = emd_loss + std_loss
        return loss
    