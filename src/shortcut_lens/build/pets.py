"""Oxford-IIIT Pet (cat vs dog) with a planted patch shortcut -- the main ground-truth testbed.

Concept: natural images at real resolution let CLIP and BLIP actually see the planted patch,
unlike small synthetic images. `P(patch | cat) = rho`, `P(patch | dog) = 1 - rho`, so the
minority (failure) groups are *cat without patch* and *dog with patch* (see D-001, PRD §7).

Pipeline position: oracle zone. Downloads via torchvision, preprocesses to a 224px JPEG cache,
then hands off to `build/planting.py` and `build/splits.py`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
