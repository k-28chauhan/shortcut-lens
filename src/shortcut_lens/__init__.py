"""shortcut-lens: find, name, verify and fix a classifier's hidden failure groups.

Concept: the package discovers "slices" (coherent clusters of errors) without ever reading
ground-truth group labels, then checks the discovery against labels only for evaluation. See
docs/PRD.md for the full pipeline and docs/ARCHITECTURE.md for the label-free/oracle boundary
that this package's subpackages are split across.

Pipeline position: this is the root package. `cli.py` is the composition root -- the only module
that may import from both the label-free and oracle zones.
"""

__version__ = "0.1.0"
