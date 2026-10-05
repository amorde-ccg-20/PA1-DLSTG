from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            future = batch.get("future_external")
            output = model(x, None if future is None else future.to(device))
            predictions.append(output.cpu().numpy())
            actuals.append(batch["y"].numpy().squeeze(-1))
    return np.concatenate(actuals), np.concatenate(predictions)


def fit(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader | None, device: torch.device,
        epochs: int, learning_rate: float, weight_decay: float, patience: int) -> tuple[nn.Module, int, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state, best_loss, wait = None, float("inf"), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = batch["x"].to(device), batch["y"].to(device).squeeze(-1)
            future = batch.get("future_external")
            loss = loss_fn(model(x, None if future is None else future.to(device)), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if valid_loader is None:
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses))})
            continue
        actual, prediction = predict(model, valid_loader, device)
        valid_loss = float(np.mean((actual - prediction) ** 2))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_mse_scaled": valid_loss})
        if valid_loss < best_loss:
            best_state, best_loss, wait = copy.deepcopy(model.state_dict()), valid_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if valid_loader is not None:
        model.load_state_dict(best_state)
    return model, epoch, history
