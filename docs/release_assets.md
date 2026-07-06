# Release Assets

Large experiment assets are published through GitHub Releases so the repository stays easy to clone while the full data bundle remains available.

## Asset Layout

The `v0.2.0` release contains three downloadable bundles:

- `patent-ipc-data-v0.2.0.zip`: cleaned patent IPC table, IPC hierarchy distributions, and field schema.
- `patent-ipc-results-v0.2.0.zip`: model comparison tables, fold metrics, and prediction caches.
- `patent-ipc-checkpoints-v0.2.0.zip`: five-fold IPCPrediction checkpoint files and related vocab/config files.

## When to Use Each Bundle

Use the data bundle if you want to inspect the corpus, label hierarchy, or class imbalance.

Use the results bundle if you want to compare model outputs without rerunning training.

Use the checkpoint bundle if you want to load the released IPCPrediction-style neural baseline.

The scripts in `experiments/` are kept in the repository because they are small and useful for review. The release assets keep larger derived files available without turning every clone into a full artifact download.
