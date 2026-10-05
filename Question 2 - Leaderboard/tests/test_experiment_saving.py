from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from scripts import run_experiment, run_sweep

PROJECT = Path(__file__).resolve().parents[1]


class ExperimentSavingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, name="a_log_target"):
        config = yaml.safe_load((PROJECT / "configs" / f"{name}.yaml").read_text())
        config["data"].update(data_dir=str(PROJECT / "Data"), seq_len=16, train_stride=128,
                              validation_origins=[536])
        config["model"].update(d_model=8, n_heads=2, e_layers=1, moving_avg=3, dropout=0.0)
        config["training"].update(epochs=1, seeds=[7], batch_size=4)
        config["output_dir"] = str(self.root / name)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path, Path(config["output_dir"])

    def run_main(self, path, *flags):
        with patch.object(sys, "argv", ["runner", "--config", str(path), "--device", "cpu", *flags]):
            run_experiment.main()

    def test_four_configs_save_results_and_refuse_overwrites(self):
        for name in ("a_log_target", "b_log_external", "c_raw_target", "d_raw_external"):
            with self.subTest(config=name):
                path, output = self.config(name)
                self.run_main(path, "--seeds", "19", "--origins", "536", "--epochs", "1")
                summary = pd.read_csv(output / "validate_summary.csv")
                self.assertEqual(len(summary), 1)
                self.assertTrue(np.isfinite(summary[["mae", "rmse", "smape"]]).all().all())
                run = output / "validate_origin536_seed19"
                for file in ("model.pt", "history.json", "metrics.json", "validation_forecast.npz"):
                    self.assertTrue((run / file).exists())
                self.assertEqual(yaml.safe_load((output / "config.yaml").read_text())["training"]["seeds"], [19])
                before = (run / "model.pt").read_bytes()
                with self.assertRaises(SystemExit):
                    self.run_main(path)
                self.assertEqual(before, (run / "model.pt").read_bytes())

    def test_failed_second_run_preserves_first_and_retry_uses_new_folder(self):
        path, output = self.config()
        original_fit = run_experiment.fit
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated failure")
            return original_fit(*args, **kwargs)
        with patch.object(run_experiment, "fit", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.run_main(path, "--seeds", "7", "19")
        self.assertEqual(len(pd.read_csv(output / "validate_summary.csv")), 1)
        self.assertTrue((output / "validate_origin536_seed7/model.pt").exists())
        retry = self.root / "retry"
        self.run_main(path, "--seeds", "19", "--output-dir", str(retry))
        self.assertTrue((retry / "validate_origin536_seed19/metrics.json").exists())
        self.assertEqual(len(pd.read_csv(output / "validate_summary.csv")), 1)

    def test_forecast_saves_predictions_and_epoch_count(self):
        path, output = self.config("d_raw_external")
        config = yaml.safe_load(path.read_text())
        config["data"]["train_stride"] = 10000
        path.write_text(yaml.safe_dump(config))
        with patch("q2_forecasting.engine.predict", side_effect=AssertionError("Fixed-budget fitting must not select a checkpoint using evaluation loss")):
            self.run_main(path, "--mode", "forecast", "--epochs", "4")
        run = output / "forecast_seed7"
        values = np.loadtxt(run / "submission.txt", delimiter=",")
        self.assertEqual(values.shape, (168,))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(json.loads((run / "metrics.json").read_text())["actual_epochs"], 4)
        history = json.loads((run / "history.json").read_text())
        self.assertEqual([entry["epoch"] for entry in history], [1, 2, 3, 4])
        self.assertTrue(all("validation_mse_scaled" not in entry for entry in history))
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["actual_epochs"], 4)

    def test_sweep_stops_on_failure(self):
        argv = ["sweep", "--device", "cpu", "--epochs", "1"]
        with patch.object(sys, "argv", argv), patch.object(run_sweep.subprocess, "run") as run:
            run_sweep.main()
        self.assertEqual(run.call_count, 4)
        self.assertIn("--epochs", run.call_args_list[0].args[0])
        with patch.object(sys, "argv", argv), patch.object(run_sweep.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "runner")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_sweep.main()
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
