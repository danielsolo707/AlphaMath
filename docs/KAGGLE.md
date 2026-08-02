# Kaggle runbook

## Inputs to attach

1. `kaggle/AlphaMath_Kaggle_Bundle.zip` built by
   `python scripts/build_kaggle_bundle.py`.
2. Qwen2.5-Math-7B-Instruct weights; point to the directory containing
   `config.json`.
3. Optional labeled benchmark (`.json`, `.jsonl`, or `.csv`).
4. Optional AIMO competition data containing `test.csv` and preferably
   `sample_submission.csv`.

Model weights are not embedded in the code ZIP.

## Automated private CLI workflow

`python scripts/build_kaggle_bundle.py` also prepares the two folders consumed
by the official Kaggle CLI:

```text
kaggle/runtime_dataset/   # metadata + generated source ZIP
kaggle/kernel/            # private GPU script + kernel metadata
kaggle/aime_benchmark_dataset/  # optional labeled AIME eval set
```

Auth: place an access token at `~/.kaggle/access_token` (or classic `kaggle.json`).

The intended sequence is:

```bash
python scripts/build_kaggle_bundle.py
# first time only:
# kaggle datasets create -p kaggle/runtime_dataset -r zip
# kaggle datasets create -p kaggle/aime_benchmark_dataset -r zip
kaggle datasets version -p kaggle/runtime_dataset -m "runtime refresh" -r zip
kaggle datasets version -p kaggle/aime_benchmark_dataset -m "aime refresh" -r zip
kaggle kernels push -p kaggle/kernel --accelerator NvidiaTeslaT4 -t 32400
kaggle kernels status danielsolo1770/alpha-math-real-model-evaluation
kaggle kernels output danielsolo1770/alpha-math-real-model-evaluation -p results/kaggle_real -o
```

The Kernel attaches:

1. `danielsolo1770/alphamath-runtime-bundle` — source + tests + config  
2. `mehedi457/qwen25-math-7b-instruct` — offline Qwen2.5-Math-7B weights  
3. `danielsolo1770/alphamath-aime-benchmark` — labeled AIME JSON (preferred)

It runs regression tests before model load, refuses unsupported GPU arches
(e.g. P100 / sm_60 with modern PyTorch), prefers `aime_2022_2024.json` when
present, and emits `alphamath_artifacts.zip` (or a diagnostic archive on failure).

**GPU note:** request Tesla T4. If Kaggle assigns P100, preflight fails loudly —
re-queue until a T4 is assigned rather than trusting default-zero scores.

## Notebook workflow

Open `notebooks/alphamath_portfolio_kaggle.ipynb` and run cells in order:

1. Configure exact paths/limits/flags when needed.
2. Extract or locate the repository.
3. Discover and validate model weights.
4. Run regression tests before paying model-load cost.
5. Load the model once and reuse it for evaluation and submission.
6. Display and download `/kaggle/working/alphamath_artifacts.zip`.

Start with `EVAL_LIMIT=3`, `SUBMISSION_LIMIT=3`, and `RUN_ABLATION=False`.
After a successful dry run, remove the limits. Enable `RUN_ABLATION=True` for
the strongest portfolio evidence; it adds two evaluation passes.

## Output contract

The final evidence archive includes:

- resolved config and environment/GPU/package manifest;
- preflight results;
- full per-attempt evaluation traces;
- flat per-problem CSV;
- Markdown report;
- optional ablation table;
- checkpointed submission and trace.

If a sample submission file is adjacent to the test data, its non-ID column is
used automatically. Otherwise `paths.answer_column` controls the output schema.

## Offline requirements

Kaggle images normally contain Torch, Transformers, NumPy, SymPy, and Accelerate.
The 7B T4 profile also requires pinned `bitsandbytes==0.49.2` because it loads
weights in 4-bit. The automated Kernel enables internet only to install that
missing wheel, records the bootstrap in its manifest/log, and still loads model
weights locally with Hugging Face offline mode enabled.
The notebook preflight checks exact availability before model loading. If a
package is missing, either enable internet temporarily to install
`requirements/gpu.txt`, or attach offline wheels as an input. Final competition
inference should use `local_files_only=true` and no external API.

## Reading the report honestly

- `backend=mock` is not a model result.
- `dataset_tier=bundled_sanity` is only an end-to-end sanity result.
- Use `dataset_tier=external_labeled` for a public accuracy claim.
- Hidden competition test data produces a submission but no local accuracy.
- Keep the artifact ZIP and manifest for every number placed in the README.
