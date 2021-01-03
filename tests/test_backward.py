from histloss import (
    HistogramLoss,
    InvHistogramLoss,
    BiHistogramLoss,
    EarthMoverDistanceLoss,
    NLLLoss
)

import pytest
import torch

@pytest.fixture(
    scope="module",
    params=[
        "HistogramLoss",
        "InvHistogramLoss",
        "BiHistogramLoss",
        "EarthMoverDistanceLoss",
        "NLLLoss"
    ],
)
def test_loss(request):
    criterion_cls = request.params
    criterion = criterion_cls()
    positive = torch.sigmoid(torch.randn(10, requires_grad=True))
    negative = torch.sigmoid(torch.randn(10, requires_grad=True))
    loss = criterion(positive, negative)
    loss.backward()
    return 0