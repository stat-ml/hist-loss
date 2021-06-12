
from histloss.nll_loss import NLLLoss
from histloss.histogram_loss import (
    HistogramLoss, InvHistogramLoss, BiHistogramLoss
)
from histloss.earth_mover_distance_loss import EarthMoverDistanceLoss
from histloss.continious_histogram_loss import ContinuousHistogramLoss
from histloss.binomial_deviance_loss import BinomialDevianceLoss

__version__ = '0.0.8'

__all__ = (
    'NLLLoss', 'HistogramLoss', 'InvHistogramLoss', 'BiHistogramLoss',
    'EarthMoverDistanceLoss', 'ContinuousHistogramLoss', 'BinomialDevianceLoss'
)