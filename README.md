# Question 2 Execution commands

Run from the repository root:

```bash
cd "Question 2 - Leaderboard"
python -m pip install -r requirements.txt
python -m scripts.smoke_test
python -m scripts.eda
python -u -m scripts.run_sweep --device cuda
python -m scripts.plot_ablation
python -u -m scripts.run_experiment --config configs/c_raw_target.yaml --mode forecast --device cuda --seeds 7 --epochs 4 --output-dir outputs/c_final_submission1
```

The EDA command summarizes the observed target's distribution and autocorrelation and checks its associations with the optional external features. It saves a short summary and plots in `Question 2 - Leaderboard/outputs/eda/`.
