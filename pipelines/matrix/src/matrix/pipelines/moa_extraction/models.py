"""Module containing the PyTorch models for the MOA extraction pipeline."""

import torch
import skorch
from torch import nn


class SkorchWrapper(skorch.NeuralNetClassifier):
    """Class to help inject Skorch NeuralNetClassifier objects through the config."""

    def __init__(
        self,
        module: str,
        optimizer: str,
        criterion: str,
        **kwargs,
    ):
        """Initialise the SkorchWrapper.

        Args:
            module: The module to use.
            optimizer: The optimizer to use.
            criterion: The criterion to use.
            **kwargs: Additional keyword arguments. For instance, module parameter may be passed as module__<param>.
        """
        super(SkorchWrapper, self).__init__(
            module=eval(module), optimizer=eval(optimizer), criterion=eval(criterion), **kwargs
        )


class TransformerBinaryClassifier(nn.Module):
    def __init__(
        self, token_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1, input_dim: int = 1
    ) -> None:
        """Initialise the TransformerBinaryClassifier.

        Args:
            token_dim: The dimension of the token embeddings.
            num_heads: The number of attention heads.
            num_layers: The number of transformer layers.
            dropout: The dropout probability.
        """
        super(TransformerBinaryClassifier, self).__init__()
        # Use Linear if input_dim provided, else LazyLinear
        self.linear = nn.Linear(input_dim, token_dim) if input_dim is not None else nn.LazyLinear(token_dim)
        # Log the input dimension once known
        self.input_dim = input_dim
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(token_dim, num_heads, dropout=dropout), num_layers
        )
        self.fc = nn.Linear(token_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TransformerBinaryClassifier.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        x = self.linear(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x  # Logit output
