import torch
from .base_hist_loss import BaseHistLoss

class ContinuousHistogramLoss(BaseHistLoss):
    """
    Histogram Loss

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
        >>> criterion = HistogramLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()

    Reference:
        CONTINUOUS HISTOGRAM LOSS: BEYOND NEURAL SIMILARITY
        https://arxiv.org/pdf/2004.02830v1.pdf
    """
    def __init__(self, bins=128, min_val=0, max_val=1, bins_similarity=3, alpha=0):
        super(BaseHistLoss, self).__init__()
        # distance
        self.bins = bins
        self.max_val = max_val
        self.min_val = min_val
        self.alpha = alpha
        
        self.delta = (self.max_val - self.min_val) / (bins - 1)
        self.t = torch.arange(self.min_val, self.max_val + self.delta, step=self.delta)

        # similarity
        self.bins_similarity = bins_similarity
        self.delta_z = 1. / (bins_similarity - 1)
        self.dz = 0.5

    def forward(self, distance, similarity):
        self.t = self.t.to(device=distance.device)
        distance = (distance - min(distance.data)) / (max(distance.data) - min(distance.data))
                
        hists = []
        std_loss = 0
        for i in range(self.bins_similarity + 1):
            mask = torch.abs(similarity / self.delta_z - i) <= self.dz
            if mask.sum() == 0:
                continue
            else:
                hist_i = self.compute_histogram(distance[mask])
                hists.append(hist_i)
                std_loss += self.std_loss(distance[mask])

        hists = torch.stack(hists) # h_rz
        phi = self.inv_cumsum_with_shift(hists) # phi_rz

        continuous_hist_loss = (hists * phi).sum() # last equation of the paper
        loss = continuous_hist_loss + std_loss
        
        return loss
    
    @staticmethod
    def inv_cumsum_with_shift(t):
        """
            phi_{rz} = sum_{q=1}^r sum_{z'=z+1}^{R_z} h_{qz'}
        """
        flip_t = torch.flip(t, [0])
        flip_cumsum = torch.cumsum(torch.cumsum(flip_t, 1), 0)
        
        cumsum = torch.flip(flip_cumsum, [0])
        zero_raw = torch.zeros_like(cumsum[-1:])
        cumsum_with_shift = torch.cat([cumsum[1:], zero_raw])
        
        return cumsum_with_shift
