# MAISProject handoff

Last updated: 2026-11-27. Branch: `feat/regressor-retrain-prep`, pull request 12.

## Goal and status

Retrain the year regressor. Code is prepared; **no training has been run** with it.

## Why

The served regressor was trained on 2,280 comments (the first 10 rows of each
monthly CSV, `max_samples_per_month=10`) for 3 epochs. Validation MAE 4.27 years,
R² 0.15. In 418 live predictions on 2026-11-27 none exceeded 10.34, so no year
after 2016 was shown. The live API reproduced the checkpoint's validation scores,
so serving and label scaling are not the cause.

## What changed

- `backend/core/regression_data.py` (new): undoes dump formatting that revealed
  the year (carriage returns, HTML escapes); drops deleted, removed, empty and
  self-declared bot comments; random sample; one copy of a repeated comment per year; month 3 held
  out for validation, month 9 for test.
- `backend/training/train_regressor.py`: 40,000 train rows per year, per-batch
  padding, output bias starts at the label mean, evaluates every 2,000 steps and
  at the last step, saves the model before evaluating, `--resume`, TF-IDF
  baseline gate, writes under `output/` by default.
- `backend/training/export_fp16.py` (new): the served half-precision copy.
- `scripts/train_all.sh`, `backend/README.md`: new commands.
- The classifiers and `backend/core/dataset.py` are unchanged.

## Verification

- Real data, no training: 760,000 train / 9,500 val / 9,500 test, all 19 years
  full, about 1 minute. No same-year comment is shared between splits.
- Mean 42 tokens per comment, 2% truncated at 256; batches pad to about 45.
- Tests (tiny random models, `Trainer.train` replaced) are kept outside the repo;
  the repo has no test folder.
- Real training steps with a tiny random model on a CPU: a run stopped mid-way
  and resumed ended with the same weights as an uninterrupted one, and its fp16
  copy loaded in `backend/core/predict.py`.
- TF-IDF baseline on the full 760,000 rows: test MAE 3.61, peak 3.7 GB, about
  6 minutes on a laptop. This is the figure the full run has to beat.
- Not verified: training `roberta-base`, GPU memory, run time, a real multi-GPU run.

## Next step

Owner: review and merge pull request 12, then run the trial and full run on a GPU
(`backend/README.md`, "Regression Model"). Deploy only if `metrics.json` has
`passes_gate: true` on the full run.

## Open points

- Compute and time figures are estimates (full run: 3-6 h on one T4).
- For `--resume` to help, `output/` must outlive the notebook.
- The CSVs hold only each month's first 20,000 comments (minutes of data in
  recent years). Resampling the raw dumps (2.23 TB compressed) is a separate
  decision.
