from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from q2_forecasting.data import prepare_data
from q2_forecasting.engine import fit, predict, set_seed
from q2_forecasting.metrics import forecast_metrics, trainable_parameters
from q2_forecasting.model import AutoformerForecaster


def model_from_config(config: dict, input_dim: int, external_dim: int) -> AutoformerForecaster:
    data, model = config["data"], config["model"]
    return AutoformerForecaster(seq_len=data["seq_len"], pred_len=data["pred_len"], input_dim=input_dim,
                                external_dim=external_dim, **model)


def loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def save_result(run_dir: Path, records: list[dict], output_root: Path, mode: str) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(records[-1], indent=2), encoding="utf-8")
    summary = output_root / f"{mode}_summary.csv"
    temporary = summary.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(summary)
    print(f"Completed {run_dir.name}: {records[-1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["validate", "forecast"], default="validate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", help="Exact output folder; must not already exist.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds; useful for rerunning an unfinished run.")
    parser.add_argument("--origins", nargs="+", type=int, help="Override validation origins.")
    parser.add_argument("--epochs", type=int, help="Override maximum epochs, for example for a timing check.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seeds is not None:
        config["training"]["seeds"] = args.seeds
    if args.origins is not None:
        if args.mode != "validate":
            parser.error("--origins is only valid in validate mode")
        config["data"]["validation_origins"] = args.origins
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if config["training"]["epochs"] < 1:
        parser.error("epochs must be positive")
    for values in (config["training"]["seeds"], config["data"]["validation_origins"]):
        if not values or len(values) != len(set(values)):
            parser.error("seeds and validation origins must be nonempty and have no duplicates")
    data_cfg, train_cfg = config["data"], config["training"]
    output_root = Path(config["output_dir"])
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output folder already exists: {output_root}. Choose a new --output-dir.")
    (output_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Saving experiment to {output_root.resolve()}", flush=True)
    device = torch.device(args.device)
    records = []
    if args.mode == "validate":
        for origin in data_cfg["validation_origins"]:
            inner_origin = origin - data_cfg["pred_len"]
            prepared = prepare_data(data_cfg["data_dir"], data_cfg["seq_len"], data_cfg["pred_len"], inner_origin,
                                    data_cfg["use_external"], data_cfg["target_transform"])
            train_set = prepared.windows(inner_origin, data_cfg["train_stride"])
            early_stop_set = prepared.windows(inner_origin + data_cfg["pred_len"], 1, origins=[inner_origin])
            outer_set = prepared.windows(origin + data_cfg["pred_len"], 1, origins=[origin])
            for seed in train_cfg["seeds"]:
                run_dir = output_root / f"validate_origin{origin}_seed{seed}"
                run_dir.mkdir(exist_ok=False)
                print(f"Starting {run_dir.name}", flush=True)
                set_seed(seed)
                model = model_from_config(config, 1 + prepared.external_dim, prepared.external_dim)
                trained, actual_epochs, history = fit(
                    model, loader(train_set, train_cfg["batch_size"], True, train_cfg["num_workers"]),
                    loader(early_stop_set, train_cfg["batch_size"], False, train_cfg["num_workers"]), device,
                    train_cfg["epochs"], train_cfg["learning_rate"], train_cfg["weight_decay"], train_cfg["patience"],
                )
                (run_dir / "history.json").write_text(json.dumps(history, indent=2))
                torch.save({"model_state": trained.state_dict(), "config": config, "seed": seed,
                            "outer_validation_origin": origin, "actual_epochs": actual_epochs}, run_dir / "model.pt")
                actual, prediction = predict(trained, loader(outer_set, train_cfg["batch_size"], False, train_cfg["num_workers"]), device)
                actual = prepared.target_scaler.inverse_transform(actual[..., None]).squeeze(-1)
                prediction = prepared.target_scaler.inverse_transform(prediction[..., None]).squeeze(-1)
                metrics = forecast_metrics(actual, prediction)
                np.savez(run_dir / "validation_forecast.npz", actual=actual, prediction=prediction)
                records.append({"origin": origin, "seed": seed, **metrics, "parameters": trainable_parameters(trained),
                                "actual_epochs": actual_epochs})
                save_result(run_dir, records, output_root, args.mode)
                del model, trained
    else:
        prepared = prepare_data(data_cfg["data_dir"], data_cfg["seq_len"], data_cfg["pred_len"], 43656,
                                data_cfg["use_external"], data_cfg["target_transform"])
        train_set = prepared.windows(43656, data_cfg["train_stride"])
        for seed in train_cfg["seeds"]:
            run_dir = output_root / f"forecast_seed{seed}"
            run_dir.mkdir(exist_ok=False)
            print(f"Starting {run_dir.name}", flush=True)
            set_seed(seed)
            model = model_from_config(config, 1 + prepared.external_dim, prepared.external_dim)
            trained, actual_epochs, history = fit(model, loader(train_set, train_cfg["batch_size"], True, train_cfg["num_workers"]),
                                                  None, device,
                                                  train_cfg["epochs"], train_cfg["learning_rate"], train_cfg["weight_decay"], train_cfg["patience"])
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save({"model_state": trained.state_dict(), "config": config, "seed": seed, "actual_epochs": actual_epochs}, run_dir / "model.pt")
            x, future = prepared.inference_window()
            trained.eval()
            with torch.no_grad():
                prediction = trained(x.to(device), None if future is None else future.to(device)).cpu().numpy()[0]
            prediction = prepared.target_scaler.inverse_transform(prediction[:, None]).squeeze(-1)
            if len(prediction) != 168 or not np.isfinite(prediction).all():
                raise RuntimeError("forecast must contain 168 finite values")
            np.savetxt(run_dir / "submission.txt", prediction[None], delimiter=",", fmt="%.10g")
            pd.DataFrame({"time_idx": np.arange(43657, 43825), "prediction": prediction}).to_csv(run_dir / "submission.csv", index=False)
            records.append({"seed": seed, "parameters": trainable_parameters(trained), "actual_epochs": actual_epochs})
            save_result(run_dir, records, output_root, args.mode)
            del model, trained


if __name__ == "__main__":
    main()
