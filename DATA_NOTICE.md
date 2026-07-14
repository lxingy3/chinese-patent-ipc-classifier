# Data notice

`data/processed/patent_ipc_clean.csv` contains public Chinese patent metadata: publication numbers, titles, abstracts, IPC codes, and fields derived from those codes. It does not contain claims, specification text, browser data, private notes, or third-party PDFs.

The table is deduplicated by normalized title and abstract. Publication numbers remain in the release so individual records can be traced to their public patent entries.

The repository's MIT license applies to the code written for this project. It does not replace the terms attached to patent metadata by its original public sources. Check those terms before redistributing the dataset or using it commercially.

The data and baseline are intended for research on patent classification, Chinese text models, and multilabel evaluation. They are not a substitute for an official patent record or professional legal review.
