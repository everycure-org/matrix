"""Module containing the PyTorch models for the MOA extraction pipeline."""

from torch import nn


class TransformerBinaryClassifier(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super(TransformerBinaryClassifier, self).__init__()
        self.linear = nn.LazyLinear(token_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(token_dim, num_heads, dropout=dropout), num_layers
        )
        self.fc = nn.Linear(token_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x  # Logit output
