"""
Utilities for PyTorch.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import GaussianBlur


def create_change_tensor(input_tensor):
    """
    Create a tensor that indicates the change of the input tensor. The first
    element of the input tensor is always considered as a change.
    Changes are indicated by 1, and no changes are indicated by 0.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len).

    Returns:
        change_tensor (torch.Tensor): Change tensor of shape (batch_size, seq_len).
    """
    change_tensor = torch.zeros_like(input_tensor).to(input_tensor.device)
    change_tensor[:, 1:] = ((input_tensor[:, 1:] != input_tensor[:, :-1]).float()).to(
        input_tensor.device
    )
    return change_tensor


def smooth_changes_gaussian(
    change_tensor: torch.Tensor, kernel_size: int = 7, sigma: float = 0.5
):
    """
    Smooth a change tensor applying a Gaussian filter.

    Args:
        change_tensor (torch.Tensor): Change tensor of shape (batch_size, seq_len).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        smooth_change_tensor (torch.Tensor): Smoothed change tensor of shape (batch_size, seq_len).
    """
    # initialize the Gaussian filter
    gaussian_filter = GaussianBlur(kernel_size=(kernel_size, 1), sigma=sigma)

    # Apply Gaussian smoothing
    smooth_change_tensor = 3 * gaussian_filter(
        change_tensor.float().unsqueeze(1)
    ).squeeze(1)

    return smooth_change_tensor.to(change_tensor.device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def remove_probabilities(
    predictions: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Remove the probabilities of the indexes in the predictions tensor that are
    not present in the target tensor.

    Args:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
        target (torch.Tensor): Target of shape (batch_size, length).

    Returns:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
    """
    # Get the unique values in the target tensor
    target_unique = target.unique(dim=-1)

    # Create a mask with the same shape as the predictions tensor
    mask = torch.zeros_like(predictions)
    # populate the mask with ones at the indexes of the unique values in the target tensor
    for i in range(target_unique.size(0)):
        mask[i, :, target_unique[i]] = 1

    # Multiply the mask with the predictions tensor
    predictions = predictions * mask
    return predictions


def smooth_probabilities(
    predictions: torch.Tensor, target: torch.Tensor, sigma: float = 0.9
) -> torch.Tensor:
    """
    Remove the probabilities of the indexes in the predictions tensor that are
    not present in the target tensor.

    Args:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
        target (torch.Tensor): Target of shape (batch_size, length).

    Returns:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
    """
    # Get the unique values in the target tensor
    target_unique = target.unique(dim=-1)

    # Create a mask with the same shape as the predictions tensor
    mask = torch.zeros_like(predictions)
    # populate the mask with ones at the indexes of the unique values in the target tensor
    for i in range(target_unique.size(0)):
        mask[i, :, target_unique[i]] = 1.3

    mask = torch.where(mask == 0, sigma, 1)

    # Multiply the mask with the predictions tensor
    predictions = predictions * mask
    return predictions


def wrong_probabilities_loss(
    predictions: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Remove the probabilities of the indexes in the predictions tensor that are
    not present in the target tensor.

    Args:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
        target (torch.Tensor): Target of shape (batch_size, length).

    Returns:
        predictions (torch.Tensor): Predictions of shape (batch_size, length, num_classes).
    """
    predictions = predictions.permute(0, 2, 1)
    predictions = torch.sigmoid(predictions)
    # Get the unique values in the target tensor
    target_unique = target.unique(dim=-1)

    # Create a mask with the same shape as the predictions tensor
    mask = torch.zeros_like(predictions)
    # populate the mask with ones at the indexes of the unique values in the target tensor
    for i in range(target_unique.size(0)):
        mask[i, :, target_unique[i]] = 1.2

    # bring everything at index 1 to 0
    mask[:, :, 0] = 0

    mask = mask * predictions

    # calculate the mse loss
    mse_loss = F.mse_loss(predictions, mask)

    return mse_loss


if __name__ == "__main__":
    # Test the function
    test = torch.tensor(
        [
            [3, 3, 3, 3, 3, 10, 10, 10, 10, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
        ]
    )

    # change = create_change_tensor(test)
    # print(change)

    # smooth_change = smooth_changes_gaussian(change, kernel_size=7, sigma=1.3)
    # print(smooth_change.round(decimals=3))

    # test the remove_probabilities function
    predictions = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.1],
                [0.3, 0.4, 0.1, 0.2],
                [0.4, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
            ],
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.1],
                [0.3, 0.4, 0.1, 0.2],
                [0.4, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
            ],
        ]
    )
    targets = torch.tensor([[1, 1, 1, 1, 1, 1], [2, 1, 0, 1, 2, 0]])

    sample_output = torch.tensor(
        [
            [
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
            ],
            [
                [0.1, 0.2, 0.3, 0.0],
                [0.2, 0.3, 0.4, 0.0],
                [0.3, 0.4, 0.1, 0.0],
                [0.4, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.3, 0.0],
            ],
        ]
    )
    # print(remove_probabilities(predictions, targets))
    print(smooth_probabilities(predictions, targets))
    # print(wrong_probabilities_loss(predictions, targets))
