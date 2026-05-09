from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "GELU": nn.GELU,
    "LeakyReLU": nn.LeakyReLU,
}


def activation_names() -> list[str]:
    return list(ACTIVATIONS.keys())


class FunctionApproximator(nn.Module):
    """A dynamic fully connected network for scalar function approximation."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_layers: int = 2,
        neurons_per_layer: int = 64,
        activation_name: str = "Tanh",
    ) -> None:
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("Hidden layer count cannot be negative.")
        if neurons_per_layer < 1:
            raise ValueError("Neurons per layer must be at least 1.")
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {activation_name}")

        layers: list[nn.Module] = []
        current_dim = input_dim
        activation_factory = ACTIVATIONS[activation_name]

        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, neurons_per_layer))
            layers.append(activation_factory())
            current_dim = neurons_per_layer

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
