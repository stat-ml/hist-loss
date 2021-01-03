import torch

def triangular_histogram_with_linear_slope(input, t, delta):
    """
    Function that calculates a histogram from an article
    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)
    Args:
        input (Tensor): tensor that contains the data
        t (Tensor): tensor that contains the nodes of the histogram
        delta (float): step in histogram
    """
    input = input.view(-1)

    # first condition of the second equation of the paper
    x = input.unsqueeze(0) - t.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    a = torch.sum(x * m, dim=1) / ( delta * len(input))

    # second condition of the second equation of the paper
    x = t.unsqueeze(0) - input.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    b = torch.sum(x * m, dim=0) / ( delta * len(input))

    return torch.add(a, b)

def inv_cumsum(t, dim=0):
    """ 
    Ordindary cumulative sum: c_i = \sum_{j=1}^{i} x_j
    Inverse cumulative sum: c_i = \sum_{j=i}^{n} x_j
    """
    flip_t = torch.flip(t, [dim])
    flip_cumsum = torch.cumsum(torch.cumsum(flip_t, dim), dim)
    inv_cumsum = torch.flip(flip_cumsum, [dim])
    return inv_cumsum
    