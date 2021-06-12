import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from histloss.utils import triangular_histogram_with_linear_slope, norm_min_max_distributuions

class BaseHistLoss(nn.Module, ABC):
    """
    Base class for all Loss with histograms

    Args:
        bins (int, optional): .Default: `128`
        alpha (float, optional): parameter for regularization. Default: `0`

    Shape:
        - pos_input: set of positive points, (N, *)
        - neg_input: set of negative points, (M, *)
        - output: scalar
    
    """
    def __init__(self, bins: int = 128, alpha: float = 0):
        super(BaseHistLoss, self).__init__()
        self.bins = bins
        self._max_val = 1
        self._min_val = 0
        self.alpha = alpha
        
        self.delta = (self._max_val - self._min_val) / (bins - 1)
        self.t = torch.arange(self._min_val, self._max_val + self.delta, step=self.delta)
    
    def compute_histogram(self, inputs: Tensor):
        return triangular_histogram_with_linear_slope(inputs, self.t, self.delta)

    @abstractmethod
    def forward(self, positive: Tensor, negative: Tensor):
        positive, negative = norm_min_max_distributuions(positive, negative)
        pass

    def std_loss(self, *inputs: Tensor):
        if self.alpha > 0:
            std_loss = self.alpha * sum(i.std() for i in inputs)
        else:
            # In order not to waste time compute unnecessary stds
            std_loss = 0 
        return std_loss
