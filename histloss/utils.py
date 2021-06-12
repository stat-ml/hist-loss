import torch
from torch import Tensor

def triangular_histogram_with_linear_slope(inputs: Tensor, t: Tensor, delta: float):
    """
    Function that calculates a histogram from an article
    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)
    Args:
        input (Tensor): tensor that contains the data
        t (Tensor): tensor that contains the nodes of the histogram
        delta (float): step in histogram
    """
    inputs = inputs.view(-1)
    # first condition of the second equation of the paper
    x = inputs.unsqueeze(0) - t.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    a = torch.sum(x * m, dim=1) / ( delta * len(inputs))

    # second condition of the second equation of the paper
    x = t.unsqueeze(0) - inputs.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    b = torch.sum(x * m, dim=0) / ( delta * len(inputs))

    return torch.add(a, b)


def norm_min_max_distributuions(*distributuions: Tensor):
    max_ = max(torch.max(d.data) for d in distributuions)
    min_ = min(torch.min(d.data) for d in distributuions)

    norm_distributuions = ((d - min_) / (max_ - min_) for d in distributuions)
    return norm_distributuions
