# Dataset fields

`data/processed/patent_ipc_clean.csv` contains one row for each unique normalized title and abstract pair. Values with more than one item use `|` as the separator.

| Field | Description |
| --- | --- |
| `patent_title` | Chinese patent title from the public source record |
| `patent_abstract` | Chinese patent abstract from the public source record |
| `publication_numbers` | Publication numbers associated with the deduplicated text |
| `ipc_codes` | All IPC codes returned for the record, with spaces removed and duplicates discarded |
| `ipc_sections` | Unique section labels derived from `ipc_codes`, such as `G` |
| `ipc_classes` | Unique class labels derived from `ipc_codes`, such as `G06` |
| `ipc_subclasses` | Unique subclass labels derived from `ipc_codes`, such as `G06F` |
| `ipc_code_count` | Number of values in `ipc_codes` |
| `ipc_level_1` | Section of the first IPC code; retained for the earlier single-label experiments |
| `ipc_level_2` | Class of the first IPC code; retained for the earlier single-label experiments |
| `ipc_level_3` | Subclass of the first IPC code; retained for the earlier single-label experiments |
| `ipc_level_4` | Full first IPC code; retained for the earlier single-label experiments |
| `ipc_level_1_name` | Chinese name for `ipc_level_1` |
| `ipc_level_2_name` | Chinese name for `ipc_level_2` |
| `ipc_level_3_name` | Chinese name for `ipc_level_3` |
| `ipc_level_4_name` | Chinese name for `ipc_level_4` |

The multilabel benchmark uses `ipc_subclasses`. It does not use the legacy `ipc_level_*` columns as prediction targets.
