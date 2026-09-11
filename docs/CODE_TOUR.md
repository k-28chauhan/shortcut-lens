# CODE TOUR — shortcut-lens

A guided reading order through the codebase. Claude Code adds a section whenever a module lands.
Each section: what the module does, the concept behind it, key functions to read, and one question
the reader should be able to answer afterwards.

Suggested order (mirrors the pipeline):

1. Foundations — `config.py`, `seeding.py`, `manifest.py`, `artifacts.py`, `cli.py`
2. The firewall — `types.py`, the import-linter contract, `build/datasets.py`, `oracle/groups.py`
3. Data — `build/planting.py`, `build/pets.py`, `build/waterbirds.py`, `build/splits.py`
4. Measurement — `metrics.py`, `stats.py`, `evaluation/core.py`
5. Models — `models/backbones.py`, `training/erm.py`
6. Embeddings — `embeddings/*`
7. Discovery — `discovery/preprocess.py`, `confidence.py`, `error_kmeans.py`, `domino_em.py`,
   `failure_direction.py`, `confirm.py`
8. Naming — `naming/*`
9. Verification — `verification/*`
10. Mitigation — `mitigation/last_layer.py`, `label_free/*`, `selection.py`, `oracle_ref/*`
11. Reporting and app — `report/*`, `viz/*`, `app/*`

---

<!-- Template for each module section:

### `path/to/module.py`
**What it does:** …
**Concept:** … (plain English, 2–4 sentences)
**Read these functions:** `a()`, `b()`
**Check yourself:** …?
-->
