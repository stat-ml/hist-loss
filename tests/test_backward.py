from histloss import (
    HistogramLoss,
    InvHistogramLoss,
    BiHistogramLoss,
    EarthMoverDistanceLoss,
    NLLLoss
)

import pytest
import torch

def _test_loss(criterion_cls):
    criterion = eval(criterion_cls)()
    positive = torch.sigmoid(torch.randn(10, requires_grad=True))
    negative = torch.sigmoid(torch.randn(10, requires_grad=True))
    loss = criterion(positive, negative)
    loss.backward()
    return 0

def test_HistogramLoss():
    return _test_loss("HistogramLoss")

def test_InvHistogramLoss():
    return _test_loss("InvHistogramLoss")

def test_BiHistogramLoss():
    return _test_loss("BiHistogramLoss")

def test_EarthMoverDistanceLoss():
    return _test_loss("EarthMoverDistanceLoss")

def test_NLLLoss():
    return _test_loss("NLLLoss")
