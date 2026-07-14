# Release assets

## v0.3.0

The v0.3.0 release has two project bundles:

- `patent-ipc-data-v0.3.0.zip` contains the enriched 9,867-document CSV, dataset statistics, field schema, and data notice.
- `patent-ipc-multilabel-results-v0.3.0.zip` contains the benchmark script, five out-of-fold result files, model card, technical report, and Python requirements.

`SHA256SUMS-v0.3.0.txt` records the SHA-256 digest of both zip files. The GitHub source archive contains the same code, tests, CI workflow, and checked-in benchmark results.

This release does not add a trained deployment checkpoint. The reference model is fitted separately inside each cross-validation fold, and the public artifact is its out-of-fold prediction file.

## v0.2.0

The v0.2.0 release remains available for the earlier single-label experiments:

- `patent-ipc-data-v0.2.0.zip` contains the earlier cleaned table, hierarchy distributions, and field schema.
- `patent-ipc-results-v0.2.0.zip` contains single-label model tables, fold metrics, and prediction caches.
- `patent-ipc-checkpoints-v0.2.0.zip` contains five ratio-attention checkpoint files with their vocabularies and configurations.

The v0.2.0 scores use the first IPC code as the target. They are not directly comparable with the multilabel F1 scores in v0.3.0.
