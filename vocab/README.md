# vocab/ — frozen files

These files are fixed before any naming result exists, so naming results cannot be tuned by editing
the word lists after seeing outputs (DECISIONS D-012).

- `generic_phrases.yaml` — phrases the `vocab` namer scores against slices (label-free zone).
- `eval_keywords.yaml` — keywords used only by evaluation to decide whether a name is correct (oracle zone).
- `FROZEN.sha256` — sha256 of both YAML files. The CLI refuses to run naming if they no longer match.

To change a file: write a DECISIONS entry, get human approval, edit, regenerate the hash with
`sha256sum generic_phrases.yaml eval_keywords.yaml > FROZEN.sha256`, and note in EXPERIMENTS.md
amendments which results were produced with the old version.
