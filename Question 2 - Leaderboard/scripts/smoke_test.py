from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2_forecasting.data import prepare_data
from q2_forecasting.model import AutoformerForecaster


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prepared = prepare_data(root / "Data", seq_len=96, pred_len=168, fit_end=43000, use_external=True, target_transform="log1p")
    batch = next(iter(torch.utils.data.DataLoader(prepared.windows(43000, 200), batch_size=2)))
    model = AutoformerForecaster(seq_len=96, pred_len=168, input_dim=11, external_dim=10, d_model=32,
                                 n_heads=4, e_layers=1, moving_avg=25, factor=2.0, dropout=0.0)
    output = model(batch["x"], batch["future_external"])
    assert output.shape == (2, 168)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    print("smoke test passed: data alignment, Autoformer output shape, and gradients")


if __name__ == "__main__":
    main()

