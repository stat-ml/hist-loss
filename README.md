
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.org/project/hist-loss/)
[![Build Status](https://travis-ci.com/stat-ml/histloss.svg?token=oPGnutpqNa9oAaMSKt7n&branch=main)](https://travis-ci.com/stat-ml/histloss)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/stat-ml/histloss/blob/main/LICENSE)

# Histogram Based Losses

This library contains implementations of some histogram-based loss functions:
- Earth Mover Distrance Loss
- Histgramm Loss ([paper](https://arxiv.org/pdf/1611.00822.pdf), [original code](https://github.com/madkn/HistogramLoss))
- Inverse Histogram Loss (our impovements)
- Bidirectinal Histogramm Loss (our impovements)
- Continuous Histogram Loss ([paper](https://arxiv.org/pdf/2004.02830v1.pdf))

Also there are implementations of another losses to compare:
- Negative Log-Likelihood
- Binomial Deviance loss ([paper](https://arxiv.org/pdf/1407.4979.pdf))

## Installation

### Installation from source
The instalation directly from this repository:
```
https://github.com/stat-ml/histloss.git
cd histloss
python setup.py install
```

### Pip Installation
```
pip install hist-loss
```


## Example of usage

```Python
>>> criterion = HistogramLoss()
>>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
>>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
>>> loss = criterion(positive, negative)
>>> loss.backward()
```
