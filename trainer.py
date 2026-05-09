from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TrainingError(RuntimeError):
    """Raised when model training fails."""


@dataclass
class PredictionSnapshot:
    epoch: int
    predictions: np.ndarray
    train_loss: float
    test_loss: float | None


@dataclass
class TrainingResult:
    model: nn.Module
    epochs: list[int]
    loss_history: list[float]
    test_loss_history: list[float | None]
    snapshots: list[PredictionSnapshot]
    stopped_early: bool = False
    normalization_stats: object | None = None


ProgressCallback = Callable[[int, int, float, float | None], None]
StopChecker = Callable[[], bool]


class Trainer:
    """Train a PyTorch model and collect prediction snapshots."""

    def __init__(
        self,
        model: nn.Module,
        train_tensors: tuple[torch.Tensor, torch.Tensor],
        test_tensors: tuple[torch.Tensor, torch.Tensor] | None,
        prediction_x: np.ndarray,
        learning_rate: float,
        optimizer_name: str,
        epochs: int,
        batch_size: int,
        snapshot_interval: int,
        device: str = "cpu",
    ) -> None:
        if epochs < 1:
            raise TrainingError("Epoch count must be at least 1.")
        if batch_size < 1:
            raise TrainingError("Batch size must be at least 1.")
        if snapshot_interval < 1:
            raise TrainingError("Snapshot interval must be at least 1.")
        if learning_rate <= 0:
            raise TrainingError("Learning rate must be positive.")
        if prediction_x.size < 2:
            raise TrainingError("Prediction grid must contain at least two x values.")

        self.model = model
        self.x_train, self.y_train = train_tensors
        self.test_tensors = test_tensors
        self.prediction_x = prediction_x
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.snapshot_interval = snapshot_interval
        self.device = torch.device(device)

        self.model.to(self.device)
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        if self.test_tensors is not None:
            self.x_test = self.test_tensors[0].to(self.device)
            self.y_test = self.test_tensors[1].to(self.device)
        else:
            self.x_test = None
            self.y_test = None

        self.prediction_tensor = torch.tensor(
            self.prediction_x.reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.model.parameters()
        if self.optimizer_name == "Adam":
            return torch.optim.Adam(parameters, lr=self.learning_rate)
        if self.optimizer_name == "SGD":
            return torch.optim.SGD(parameters, lr=self.learning_rate)
        if self.optimizer_name == "RMSprop":
            return torch.optim.RMSprop(parameters, lr=self.learning_rate)
        raise TrainingError(f"Unsupported optimizer: {self.optimizer_name}")

    def _evaluate_loss(
        self,
        criterion: nn.Module,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_tensor)
            loss = criterion(predictions, y_tensor)
        return float(loss.detach().cpu().item())

    def _test_loss(self, criterion: nn.Module) -> float | None:
        if self.x_test is None or self.y_test is None or self.x_test.numel() == 0:
            return None
        return self._evaluate_loss(criterion, self.x_test, self.y_test)

    def _make_snapshot(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float | None,
    ) -> PredictionSnapshot:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.prediction_tensor)
        prediction_values = predictions.detach().cpu().numpy().reshape(-1)
        return PredictionSnapshot(
            epoch=epoch,
            predictions=prediction_values,
            train_loss=train_loss,
            test_loss=test_loss,
        )

    def train(
        self,
        progress_callback: ProgressCallback | None = None,
        stop_checker: StopChecker | None = None,
    ) -> TrainingResult:
        criterion = nn.MSELoss()
        optimizer = self._make_optimizer()
        dataset = TensorDataset(self.x_train, self.y_train)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        epochs: list[int] = []
        train_history: list[float] = []
        test_history: list[float | None] = []
        snapshots: list[PredictionSnapshot] = []

        initial_train_loss = self._evaluate_loss(criterion, self.x_train, self.y_train)
        initial_test_loss = self._test_loss(criterion)
        snapshots.append(self._make_snapshot(0, initial_train_loss, initial_test_loss))

        stopped_early = False
        for epoch in range(1, self.epochs + 1):
            if stop_checker is not None and stop_checker():
                stopped_early = True
                break

            self.model.train()
            running_loss = 0.0
            sample_count = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                running_loss += float(loss.detach().cpu().item()) * batch_size
                sample_count += batch_size

            train_loss = running_loss / max(sample_count, 1)
            test_loss = self._test_loss(criterion)

            if not np.isfinite(train_loss) or (
                test_loss is not None and not np.isfinite(test_loss)
            ):
                raise TrainingError(
                    "Training became unstable. Try a lower learning rate or a different activation."
                )

            epochs.append(epoch)
            train_history.append(train_loss)
            test_history.append(test_loss)

            if (
                epoch % self.snapshot_interval == 0
                or epoch == self.epochs
            ):
                snapshots.append(self._make_snapshot(epoch, train_loss, test_loss))

            if progress_callback is not None:
                progress_callback(epoch, self.epochs, train_loss, test_loss)

        if train_history and snapshots[-1].epoch != epochs[-1]:
            snapshots.append(
                self._make_snapshot(epochs[-1], train_history[-1], test_history[-1])
            )

        self.model.to("cpu")
        return TrainingResult(
            model=self.model,
            epochs=epochs,
            loss_history=train_history,
            test_loss_history=test_history,
            snapshots=snapshots,
            stopped_early=stopped_early,
        )
