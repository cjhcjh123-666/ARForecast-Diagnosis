"""Training loop shared by all first-milestone models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.metrics import forecast_metrics


def _free_running_mse(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for context, future in loader:
            context = context.to(device)
            future = future.to(device)
            prediction = model.predict(context, future.size(1))
            losses.append(torch.mean((prediction - future) ** 2).item())
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> list[dict[str, float]]:
    """Optimize teacher-forced objectives and monitor free-running validation MSE."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for context, future in train_loader:
            context = context.to(device)
            future = future.to(device)
            loss = model.training_loss(context, future)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
        val_mse = _free_running_mse(model, val_loader, device)
        row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_free_mse": val_mse}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_free_mse={row['val_free_mse']:.6f}"
        )
    return history


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Collect free-running predictions and metrics."""

    model.eval()
    predictions, targets, contexts = [], [], []
    for context, future in loader:
        context_device = context.to(device)
        prediction = model.predict(context_device, future.size(1)).cpu().numpy()
        predictions.append(prediction)
        targets.append(future.numpy())
        contexts.append(context.numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    context_array = np.concatenate(contexts, axis=0)
    return forecast_metrics(prediction_array, target_array), prediction_array, target_array, context_array
