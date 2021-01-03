import torch
from .base_hist_loss import BaseHistLoss

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
    def __init__(self, bins=128, min_val=-1, max_val=1, alpha=0, method='sim'):
        super(EarthMoverDistanceLoss, self).__init__(bins, min_val, max_val, alpha)
        if method not in {'sim', 'asim'}:
            raise NotImplementedError(f'Undefined method of EMD Loss: {method}')
        self.method = method

    def forward(self, positive, negative):
        self.t = self.t.to(device=positive.device)
        
        pos_hist = self.compute_histogram(positive) # h_pos
        neg_hist = self.compute_histogram(negative) # h_neg
        
        if self.method == 'sim':
            emd_loss = - (torch.abs(torch.cumsum(neg_hist - pos_hist, 0))).sum()
        elif self.method == 'asim':
            emd_loss = (torch.cumsum(pos_hist - neg_hist, 0)).sum()

        std_loss = self.std_loss(positive, negative)

        loss = emd_loss + std_loss
        return loss
    