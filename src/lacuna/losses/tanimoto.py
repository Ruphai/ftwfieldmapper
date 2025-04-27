import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace

class BinaryTanimotoDualLoss(nn.Module):
    """
    Computes the Binary Tanimoto Dual Loss between predictions and reference annotations.

    The loss is computed as the average of the Tanimoto loss for the original predictions
    and their inverted counterparts. The Tanimoto loss is a similarity measure between
    two sets and is particularly useful for imbalanced datasets.

    Attributes:
        smooth (float): A small constant added to the numerator and denominator to
                        prevent division by zero and stabilize the loss.

    Methods:
        _binary_tanimoto_loss(predictions, ref_annotation): Computes the Tanimoto loss
                                                            between predictions and
                                                            reference annotations.
        forward(predictions, ref_annotation): Computes the Binary Tanimoto Dual Loss.

    Usage:
        loss_function = BinaryTanimotoDualLoss(smooth=1e-5)
        loss = loss_function(predictions, labels)

    Args:
        smooth (float, optional): Smoothing factor to prevent division by zero.
                                  Default is 1.0e-5.

    Returns:
        torch.Tensor: A scalar tensor containing the Binary Tanimoto Dual Loss.

    Note:
        The Tanimoto loss function, also known as the Jaccard loss function, is
        a type of fuzzy loss function that measures the similarity between two sets.
        It is particularly useful for tasks such as image segmentation where the
        degree of uncertainty in the predicted segmentation mask needs to be taken
        into account.

        Mathematically, the Tanimoto loss is defined as:

            L(p, l) = sum(p_i * l_i) / (sum(p_i^2 + l_i^2) - sum(p_i * l_i))

        Where:
            p = {p_i}, p_i ∈ [0,1] is a continuous variable representing the vector
                of probabilities for the i-th pixel.
            l = {l_i} are the corresponding ground truth labels. For binary
                vectors, l_i ∈ {0,1}.
            The batch size is considered as the first dimension.

        To achieve faster training convergence, a dual form of the Tanimoto loss
        is introduced. This dual form measures the overlap area of the complement
        of the regions of interest. Specifically, if p_i measures the probability
        of the ith pixel to belong to class l_i, the complement loss is defined as:

            D(1−p_i, 1−l_i)

        Where the subtraction is performed element-wise. The intuition behind
        using the complement in the loss function comes from the fact that the
        numerator of the Dice coefficient, sum(p_i * l_i), can be viewed as an
        inner product between the probability vector, p, and the ground truth
        label vector, l. The part of the probabilities vector, p_i, that corresponds
        to the elements of the label vector, l_i, that have zero entries, does not
        alter the value of the inner product.
    """
    def __init__(self, smooth=1.0e-5):
        super(BinaryTanimotoDualLoss, self).__init__()
        self.smooth = smooth

    def _binary_tanimoto_loss(self, predictions, ref_annotation):
        
        ref_annotation_squared_sum = (ref_annotation * ref_annotation).sum(dim=1)
        predictions_squared_sum = (predictions * predictions).sum(dim=1)
        intersection = (ref_annotation * predictions).sum(dim=1)

        denominator = ref_annotation_squared_sum + predictions_squared_sum - intersection
        loss = (intersection + self.smooth) / (denominator + self.smooth)

        return loss

    def forward(self, predictions, ref_annotation):
        assert predictions.shape[0] == ref_annotation.shape[0], (
        f" Dimension zero mismatch between 'Predictions' shape "
        f"[{predictions.shape}], and 'ref_annotation' shape"
        f"[{ref_annotation.shape}].")

        ref_annotation = ref_annotation.float().contiguous().view(ref_annotation.shape[0], -1)
        predictions = predictions.contiguous().view(predictions.shape[0], -1)

        primary_loss = self._binary_tanimoto_loss(predictions, ref_annotation)
        inverted_predictions = 1.0 - predictions
        inverted_ref_annotation = 1.0 - ref_annotation
        secondary_loss = self._binary_tanimoto_loss(inverted_predictions, inverted_ref_annotation)

        combined_loss = 0.5 * (primary_loss + secondary_loss)

        return (1 - combined_loss).mean()


class TanimotoDualLoss(nn.Module):
    """
    Computes the multi-class Tanimoto Dual Loss between predictions and reference annotations.

    This loss is an extension of the Binary Tanimoto Dual Loss to handle multi-class scenarios.
    For each class, the Binary Tanimoto Dual Loss is computed and then weighted and summed
    to get the final loss value.

    Attributes:
        weight (torch.Tensor or list, optional): A tensor or list containing the weights
                                                 for each class. If not provided, weights
                                                 are uniformly set based on the number of classes.
        ignore_index (int, optional): Specifies a class index to be ignored during the loss
                                      computation. Default is -100.
        smooth (float, optional): Smoothing factor used in the Binary Tanimoto Dual Loss
                                  to prevent division by zero. Default is 1.0e-5.

    Methods:
        forward(predictions, ref_annotation): Computes the multi-class Tanimoto Dual Loss.

    Usage:
        loss_function = TanimotoDualLoss(weight=[0.5, 0.3, 0.2], ignore_index=2)
        loss = loss_function(predictions, labels)

    Args:
        weight (torch.Tensor or list, optional): Weights for each class. Default is None.
        ignore_index (int, optional): Class index to ignore. Default is -100.
        smooth (float, optional): Smoothing factor for Binary Tanimoto Dual Loss.
                                  Default is 1.0e-5.

    Returns:
        torch.Tensor: A scalar tensor containing the multi-class Tanimoto Dual Loss.
    """
    def __init__(self, weight=None, ignore_index=-100, smooth=1.0e-5):
        super(TanimotoDualLoss, self).__init__()
        self.binary_tanimoto_dual_loss = BinaryTanimotoDualLoss(smooth=smooth)
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predictions, ref_annotation):
        device = predictions.device
        nclass = predictions.shape[1]

        # Convert target annotations to one-hot encoding if they aren't already
        if predictions.shape == ref_annotation.shape:
            pass
        elif (len(predictions.shape) == 4) and (predictions.shape != ref_annotation.shape):
            ref_annotation = F.one_hot(ref_annotation, 
                                       num_classes=nclass).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("The shapes of 'predictions' and 'ref_annotation' are incompatible.")
            

        total_loss = 0

        # Handling weights
        if self.weight is None:
            self.weight = torch.Tensor([1./nclass] * nclass).to(device)
        else:
            if isinstance(self.weight, list):
                self.weight = torch.tensor(self.weight, dtype=torch.float32).to(device)

        predictions = F.softmax(predictions, dim=1)

        for i in range(nclass):
            if i != self.ignore_index:
                tanimoto_loss = self.binary_tanimoto_dual_loss(predictions[:, i], ref_annotation[:, i])
                assert self.weight.shape[0] == nclass, \
                    'Expect weight shape [{}], get[{}]'\
                    .format(nclass, self.weight.shape[0])
                tanimoto_loss *= self.weight[i]
                total_loss += tanimoto_loss

        return total_loss

# class LocallyWeightedTanimotoDualLoss(TanimotoDualLoss):
#     """
#     Tanimoto Dual Loss weighted by inverse of label frequency.
#     Inherits from TanimotoDualLoss to handle multi-class scenarios with locally computed weights.
#     """
#     def __init__(self, ignore_index=-100, smooth=1.0e-5):
#         # Initialize the parent class with the smooth factor and ignore index.
#         # Weight is not initialized here as it will be computed for each batch in the forward method.
#         super(LocallyWeightedTanimotoDualLoss, self).__init__(smooth=smooth, ignore_index=ignore_index)

#     def calculate_weights(self, ref_annotation, nclass):
#         """
#         Calculate inverse frequency weights for each class based on the input batch.
#         """
#         unique, unique_counts = torch.unique(ref_annotation, return_counts=True)
#         # Handle ignore index
#         unique_counts = unique_counts[unique != self.ignore_index]
#         unique = unique[unique != self.ignore_index]
        
#         ratio = unique_counts.float() / torch.numel(ref_annotation)
#         weight = (1. / ratio) / torch.sum(1. / ratio)

#         loss_weight = torch.ones(nclass, device=ref_annotation.device) * 0.00001
#         for i in range(len(unique)):
#             loss_weight[unique[i]] = weight[i]
            
#         return loss_weight

#     def forward(self, predictions, ref_annotation):
#         """
#         Computes the weighted Tanimoto Dual Loss for the given predictions and reference annotations.
#         """
#         nclass = predictions.shape[1]

#         # Calculate the weights based on the current batch
#         loss_weight = self.calculate_weights(ref_annotation, nclass)
        
#         # Use the parent class's forward method, passing in the locally calculated weights
#         return super().forward(predictions, ref_annotation, weight=loss_weight)


# class LocallyWeightedTanimotoDualLoss(nn.Module):
#     """
#     Tanimoto Dual Loss weighted by inverse of label frequency.
#     ...

#     Methods:
#         calculate_weights(ref_annotation): Calculate inverse frequency weights
#                                            for each class based on the input batch.
#     """
#     def __init__(self, ignore_index=-100, smooth=1.0e-5):
#         super(LocallyWeightedTanimotoDualLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth

#     def calculate_weights(self, ref_annotation, nclass):
#         unique, unique_counts = torch.unique(ref_annotation, return_counts=True)
#         # Handle ignore index
#         unique_counts = unique_counts[unique != self.ignore_index]
#         unique = unique[unique != self.ignore_index]
        
#         ratio = unique_counts.float() / torch.numel(ref_annotation)
#         weight = (1. / ratio) / torch.sum(1. / ratio)

#         loss_weight = torch.ones(nclass, device=ref_annotation.device) * 0.00001
#         for i in range(len(unique)):
#             loss_weight[unique[i]] = weight[i]
            
#         return loss_weight

#     def forward(self, predictions, ref_annotation):
#         device = predictions.device
#         nclass = predictions.shape[1]

#         # Calculate the weights based on the current batch
#         loss_weight = self.calculate_weights(ref_annotation, nclass)

#         # Use TanimotoDualLoss with the calculated weights
#         tanimoto_dual_loss = TanimotoDualLoss(weight=loss_weight, ignore_index=self.ignore_index, 
#                                               smooth=self.smooth)
        
#         # Compute the loss
#         total_loss = tanimoto_dual_loss(predictions, ref_annotation)

#         return total_loss


class LocallyWeightedTanimotoDualLoss(nn.Module):
    """
    Tanimoto Dual Loss weighted by inverse of label frequency.
    ...

    Methods:
        calculate_weights(ref_annotation): Calculate inverse frequency weights
                                           for each class based on the input batch.
    """
    def __init__(self, ignore_index=-100, smooth=1.0e-5):
        super(LocallyWeightedTanimotoDualLoss, self).__init__()
        self.ignore_index = ignore_index
        self.tanimoto_dual_loss = TanimotoDualLoss(ignore_index=self.ignore_index, smooth=smooth)

    def calculate_weights(self, ref_annotation, nclass):

        unique, unique_counts = torch.unique(ref_annotation, return_counts=True)
        # Handle ignore index
        unique_counts = unique_counts[unique != self.ignore_index]
        unique = unique[unique != self.ignore_index]
        
        ratio = unique_counts.float() / torch.numel(ref_annotation)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        loss_weight = torch.ones(nclass, device=ref_annotation.device) * 0.00001
        for i in range(len(unique)):
            loss_weight[unique[i]] = weight[i]
            
        return loss_weight

    def forward(self, predictions, ref_annotation):

        device = predictions.device
        nclass = predictions.shape[1]

        # Calculate the weights based on the current batch
        loss_weight = self.calculate_weights(ref_annotation, nclass)
        
        self.tanimoto_dual_loss.weight = loss_weight
        total_loss = self.tanimoto_dual_loss(predictions, ref_annotation)

        return total_loss
