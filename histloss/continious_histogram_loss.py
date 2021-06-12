import torch
from torch import Tensor
from histloss.base_hist_loss import BaseHistLoss
from histloss.utils import norm_min_max_distributuions

class ContinuousHistogramLoss(BaseHistLoss):
    """
    Histogram Loss

    Args:
        bins (int, optional): .Default: `10`
        alpha (float, optional): parameter for regularization. Default: `0`

    Shape:
        - distance: contain predicted distance from model, (N, *)
        - similarity: contain real distance from data, (N, *)
        - loss: scalar

    Examples::
        >>> criterion = ContinuousHistogramLoss()
        >>> distance = torch.rand(100, requires_grad=True)
        >>> similarity = torch.randint(low=0, high=5, size=(100,)).to(torch.float)
        >>> loss = criterion(distance, similarity)
        >>> loss.backward()

    Reference:
        CONTINUOUS HISTOGRAM LOSS: BEYOND NEURAL SIMILARITY
        https://arxiv.org/pdf/2004.02830v1.pdf
    """
    def __init__(self, bins: int = 128, bins_similarity: int = 3, alpha: float = 0):
        super(ContinuousHistogramLoss, self).__init__(bins=bins, alpha=alpha)

        # similarity
        if bins_similarity < 1:
            raise ValueError(
                f'Number of bins for similarity must be grather than 1: {bins_similarity}'
            )
        
        self.bins_similarity = bins_similarity
        self.delta_z = 1. / (bins_similarity - 1)
        self.dz = 0.5

    def forward(self, distance: Tensor, similarity: Tensor):
        self.t = self.t.to(device=distance.device)
        distance, = norm_min_max_distributuions(distance)
                
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
