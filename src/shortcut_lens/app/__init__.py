"""Label-free zone: Gradio slice explorer. Reads only a precomputed demo bundle.

Concept: the app never imports oracle-zone modules -- it reads bundle files exported by
`slens export-demo`, which is itself in the oracle zone (bundle creation may use ground truth to
decide what to include, but the app serving it only reads files).

Pipeline position: label-free zone (see docs/ARCHITECTURE.md §1).
"""
