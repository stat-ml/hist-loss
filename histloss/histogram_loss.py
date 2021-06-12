import torch
from torch import Tensor
from histloss.base_hist_loss import BaseHistLoss
from histloss.utils import norm_min_max_distributuions

class HistogramLoss(BaseHistLoss):
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
        E. Ustinova and V. Lempitsky: Learning Deep Embeddings with Histogram Loss:
        https://arxiv.org/pdf/1611.00822.pdf
    """
    def forward(self, positive: Tensor, negative: Tensor):
        self.t = self.t.to(device=positive.device)
        positive, negative = norm_min_max_distributuions(positive, negative)
        
        pos_hist = self.compute_histogram(positive) # h_pos
        neg_hist = self.compute_histogram(negative) # h_neg
        pos_cum = torch.cumsum(pos_hist, 0) # phi_pos

        hist_loss = (neg_hist * pos_cum).sum() # 4 equation of the paper
        # Not in the article, own improvements
        std_loss = self.std_loss(positive, negative)

        loss = hist_loss + std_loss
        return loss


class InvHistogramLoss(BaseHistLoss):
    """
    Inverse Histogram Loss

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
        >>> criterion = InvHistogramLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()
    """
    def forward(self, positive: Tensor, negative: Tensor):
        self.t = self.t.to(device=positive.device)
        positive, negative = norm_min_max_distributuions(positive, negative)

        pos_hist = self.compute_histogram(positive)
        neg_hist = self.compute_histogram(negative)
        neg_inv_cum = neg_hist.flip(0).cumsum(0).flip(0)

        inv_hist_loss = (pos_hist * neg_inv_cum).sum()
        std_loss = self.std_loss(positive, negative)

        loss = inv_hist_loss + std_loss
        return loss


class BiHistogramLoss(BaseHistLoss):
    """
    Biderctional Histogram Loss

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
        >>> criterion = BiHistogramLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()
    """
    def forward(self, positive: Tensor, negative: Tensor):
        self.t = self.t.to(device=positive.device)
        positive, negative = norm_min_max_distributuions(positive, negative)
        
        pos_hist = self.compute_histogram(positive) # h_pos
        pos_cum = torch.cumsum(pos_hist, 0)
        neg_hist = self.compute_histogram(negative) # h_neg
        neg_inv_cum = neg_hist.flip(0).cumsum(0).flip(0)

        hist_loss = (neg_hist * pos_cum).sum()
        inv_hist_loss = (pos_hist * neg_inv_cum).sum()
        std_loss = self.std_loss(positive, negative)
        
        loss = hist_loss + inv_hist_loss + std_loss
        return loss
