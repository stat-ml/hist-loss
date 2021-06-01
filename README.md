[![Build Status](https://travis-ci.com/stat-ml/histloss.svg?token=oPGnutpqNa9oAaMSKt7n&branch=main)](https://travis-ci.com/stat-ml/histloss)

# Histogram Based Losses

This library contains implementations of some loss functions:
- Negative Log-Likelihood
- Earth Mover Distrance Loss
- Histgramm Loss ([paper](https://arxiv.org/pdf/1611.00822.pdf), [original code](https://github.com/madkn/HistogramLoss))
- Binomial Deviance loss ([paper](https://arxiv.org/pdf/1407.4979.pdf))
- Inverse Histogram Loss (our impovements)
- Bidirectinal Histogramm Loss (our impovements)
- Continuous Histogram Loss [paper](https://arxiv.org/pdf/2004.02830v1.pdf)

## Example of usage

```Python
>>> criterion = HistogramLoss()
>>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
>>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
>>> loss = criterion(positive, negative)
>>> loss.backward()
```
