import torch
from torch import nn
import torch.nn.functional as F

class LocallyWeightedEntropyLoss(nn.Module):
    """
    Cross Entropy loss weighted by inverse of label frequency calculated locally based
    on the input batch.

    Args:

        ignore_index (int): Class index to ignore
        reduction (str): Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'

    Returns:

        Loss tensor according to arg reduction

    """

    def __init__(self, ignore_index=-100, reduction='mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def calculate_weights(self, target, num_class):

        unique, unique_counts = torch.unique(target, return_counts=True)
        # calculate weight for only valid indices
        unique_counts = unique_counts[unique != self.ignore_index]
        unique = unique[unique != self.ignore_index]
        ratio = unique_counts.float() / torch.numel(target)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        loss_weight = torch.ones(num_class, device=target.device) * 0.00001
        for i in range(len(unique)):
            loss_weight[unique[i]] = weight[i]

        return loss_weight
    def forward(self, predict, target):
        num_class = predict.shape[1]
        # Calculate the weights based on the current batch
        loss_weight = self.calculate_weights(target, num_class)

        # Initialize the loss
        loss = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss(predict, target)
