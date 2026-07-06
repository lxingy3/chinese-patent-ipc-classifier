# Technical Report

This project studies supervised IPC classification for Chinese patent titles and abstracts. The release is built around a cleaned public patent metadata table, model comparison outputs, prediction caches, and scripts that make the main experiments inspectable without requiring a full GPU rerun.

## Data Source

The source records are public patent metadata collected for patent-domain analysis. Each record contains a patent title and abstract, and the IPC labels were completed from public patent web pages where possible. IPC names were resolved from the 2026.01 IPC classification tables.

The raw collection contained 10,000 patent records. After removing records without IPC codes and deduplicating by `title + abstract`, the supervised training table contains 9,867 usable samples.

The released table keeps the fields needed for reproducible classification:

- patent title and abstract
- parsed IPC level 1, level 2, level 3, and level 4 labels
- label distribution tables for each IPC level
- prediction outputs from the released model comparisons

For the supervised experiments, the first IPC code is used as the main single-label target. This keeps the evaluation protocol simple and comparable across models, but it also means the task does not fully represent the original multi-label nature of IPC assignment.

## Preprocessing

The preprocessing pipeline does four main things.

First, it normalizes patent text fields. Empty values are handled consistently, whitespace is collapsed, and the title and abstract are combined as the model input.

Second, it parses IPC hierarchy levels from the main IPC code. The project evaluates level 1, level 2, and level 3 labels. Level 4 is kept in the dataset, but it is too sparse for stable single-label training in this corpus.

Third, it removes unsuitable training rows. Records without IPC labels are dropped, and duplicate `title + abstract` pairs are removed to avoid repeated samples leaking across folds.

Fourth, it analyzes label sparsity before modeling. The final dataset has 8 level-1 classes, 114 level-2 classes, 458 level-3 classes, and 4,485 level-4 classes. At level 3, 40.6% of classes have fewer than 5 samples. At level 4, the median class has only one sample, so level-4 classification is not treated as a reliable training target in this release.

## Model Principles

### TF-IDF + Linear SVM

The strongest lightweight baseline uses character-level TF-IDF with a linear SVM. Character n-grams work well for Chinese patent text because many technical phrases, such as control method, image processing, battery management, and detection device, can be captured without depending on a tokenizer.

The title is repeated before appending the abstract. This gives more weight to the title, which usually contains a compact description of the invention. The best released SVM configuration uses:

- `C=4`
- character n-grams from 1 to 4
- `title_weight=10`
- up to 200,000 TF-IDF features

Each fold fits the vectorizer only on the training split, then transforms the validation split. This avoids vocabulary leakage from validation data.

### IPCPrediction-Style Ratio-Attention Model

The neural baseline follows the idea of IPCPrediction: represent each patent as a set of technical terms, weight those terms by relative importance, and decode IPC labels through a hierarchy-aware classifier.

The original IPCPrediction paper uses patent claims. This dataset only provides titles and abstracts, so the implementation approximates technical terms with character n-grams, alphanumeric fragments, and weighted text features. Each term is embedded, multiplied by a ratio-style weight, passed through attention layers, and decoded into IPC levels.

This model is useful as a structured neural baseline, but the released results show that it needs richer patent text or more data to beat the sparse linear baseline.

### Chinese-XLNet Fine-Tuning

The transformer experiment fine-tunes `hfl/chinese-xlnet-mid` for IPC level-3 classification and derives level-1 and level-2 predictions from the predicted level-3 code. The model uses the title and abstract as input, tokenizes with the XLNet tokenizer, and trains a sequence-classification head over 458 level-3 labels.

The full five-fold run was performed in a GPU environment. The repository publishes the metric summaries and training entrypoint so the result can be inspected without repeating the full GPU run.

## Code Design

The repository separates lightweight reproducibility from full experiment recovery.

- `experiments/baseline_tfidf_svm.py` runs a small local SVM baseline.
- `experiments/tfidf_svm_cv.py` runs cross-validation for TF-IDF/SVM settings and writes prediction caches.
- `experiments/ratio_attention_ipc.py` contains the PyTorch ratio-attention IPC model and checkpoint-compatible code.
- `experiments/chinese_xlnet_oof.py` is the optional GPU entrypoint for Chinese-XLNet five-fold evaluation.
- `scripts/validate_release.py` checks the release structure, required files, and artifact readability.

The published checkpoint directory contains a five-fold IPCPrediction checkpoint set. The prediction caches are included so readers can audit per-sample outputs and compare models without rerunning every training job.

## Result Analysis

The best lightweight model is TF-IDF + Linear SVM with `C=4`, character n-grams `1-4`, and `title_weight=10`. Its five-fold out-of-fold accuracies are:

| Target | Accuracy |
| --- | ---: |
| IPC level 1 | 0.7258 |
| IPC level 2 | 0.5818 |
| IPC level 3 | 0.4656 |

Chinese-XLNet improves the best level-3 result, but at a much higher compute cost:

| Target | Accuracy |
| --- | ---: |
| IPC level 1 | 0.7561 |
| IPC level 2 | 0.6016 |
| IPC level 3 | 0.4875 |

The IPCPrediction-style model is weaker on this dataset. The best released 16-epoch run reaches 0.5502 at level 1, 0.2688 at level 2, and 0.1569 at level 3.

The main reason is data shape. IPCPrediction-style models are designed for richer patent claims and larger corpora. With fewer than 10,000 title-and-abstract samples and hundreds of level-3 labels, the neural model struggles with long-tail classes. The TF-IDF/SVM baseline is less expressive, but it is very strong for medium-sized short-text classification because the sparse character n-grams capture repeated technical phrases reliably.

Chinese-XLNet confirms that pretrained language models can add useful semantic information. Its gain over SVM is real but moderate, and the training cost is much higher. For practical use, SVM is the best local baseline; for accuracy-focused experiments, pretrained transformers are the better direction.

## Limitations

The dataset uses public patent metadata, but it does not include full claims or specification text. Some patent categories need details that are not visible from the title and abstract alone.

The supervised setup converts IPC labels to a single main label. This simplifies evaluation, but it loses secondary IPC assignments and makes the task less faithful to real patent classification.

The label distribution is heavily imbalanced. Level-3 classification already has many low-support labels, and level-4 classification is not reliable at this dataset size.

The released Chinese-XLNet results are published as summary artifacts because the full five-fold run is GPU-heavy. The script is included, but exact runtime depends on hardware and library versions.

## Future Work

The most useful next step is multi-label IPC prediction, where each patent can keep all assigned IPC codes instead of only the first one.

Other improvements worth testing:

- hierarchical losses that jointly optimize level 1, level 2, and level 3
- label embeddings or IPC-tree constraints to reduce impossible predictions
- patent-domain Chinese pretrained models
- longer input text from claims or specification sections
- calibration and confidence reporting for low-support labels
- external validation on another public patent collection

The current release is meant to be a clean, inspectable benchmark for this patent corpus rather than a final patent classification system.
