"""Segmentation loss functions.

The notebook this was ported from also defined a `DiceBCELoss` and a
`WeightedHausdorffDistance`, but neither is usable as written: the former
detaches its output from the autograd graph (backprop through it is a
no-op), and the latter calls undefined helpers (`cartesian`, `cdist`,
`generaliz_mean`, `_assert_no_grad`) that were never carried over from
wherever that implementation was copied from. Both were unused (commented
out) in the notebook, so they were left out here rather than ported broken.
"""

import torch.nn as nn


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class ComboLoss(nn.Module):
    """Weighted combination of Dice loss and a modified cross-entropy term.

    alpha < 0.5 penalises false positives more, > 0.5 penalises false negatives more.
    ce_ratio weights the cross-entropy term against the Dice term.
    """

    def __init__(self, alpha=0.5, ce_ratio=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def forward(self, inputs, targets, smooth=1, eps=1e-9):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = inputs.clamp(eps, 1.0 - eps)
        ce = -(
            self.alpha * targets * inputs.log()
            + (1 - self.alpha) * (1.0 - targets) * (1.0 - inputs).log()
        )
        weighted_ce = ce.mean(-1)

        return self.ce_ratio * weighted_ce - (1 - self.ce_ratio) * dice
