import torch
from abc import *
from .utils import triangular_histogram_with_linear_slope

class BaseHistLoss(torch.nn.Module, ABC):
    """
    Base class for all Loss with histograms

    Args:
        bins (int, optional): .Default: `128`
        min_val (float, optional): Default: `-1`
        max_val (float, optional): Default: `1`
        alpha (float, optional): parameter for regularization. Default: `0`

    Shape:
        - pos_input: set of positive points, (N, *)
        - neg_input: set of negative points, (M, *)
        - output: scalar
    
    """
    def __init__(self, bins=128, min_val=-1, max_val=1, alpha=0):
        super(BaseHistLoss, self).__init__()
        self.bins = bins
        self.max_val = max_val
        self.min_val = min_val
        self.alpha = alpha
        
        self.delta = (self.max_val - self.min_val) / (bins - 1)
        self.t = torch.arange(self.min_val, self.max_val + self.delta, step=self.delta)
    
    def compute_histogram(self, inputs):
        return triangular_histogram_with_linear_slope(inputs, self.t, self.delta)

    @abstractmethod
    def forward(self, positive, negative):
        pass

    def std_loss(self, *inputs):
        if self.alpha > 0:
            std_loss = self.alpha * sum(i.std() for i in inputs)
        else:
            # In order not to waste time compute unnecessary stds
            std_loss = 0 
        return std_loss