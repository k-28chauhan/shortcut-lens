"""Contract test: vocab/*.yaml must match the recorded FROZEN.sha256 (D-012).

If this fails, someone edited a frozen vocabulary file. Do not "fix" it by regenerating the hash --
see vocab/README.md for the real process (a DECISIONS entry and human approval come first).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_DIR = REPO_ROOT / "vocab"
FROZEN_FILE = VOCAB_DIR / "FROZEN.sha256"

_LINE_RE = re.compile(r"^([0-9a-f]{64})\s+\*?(.+)$")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_frozen_file(path: Path) -> dict[str, str]:
    """Parse `sha256sum`-format lines (`"<hash>  <filename>"`) into `{filename: hash}`."""
    entries: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            raise ValueError(f"Unparseable line in {path}: {line!r}")
        digest, filename = match.groups()
        entries[filename] = digest
    return entries


def test_vocab_files_match_frozen_hash() -> None:
    expected = _parse_frozen_file(FROZEN_FILE)
    assert expected, "FROZEN.sha256 is empty or unparseable"

    for filename, expected_digest in expected.items():
        actual_digest = _sha256_of(VOCAB_DIR / filename)
        assert actual_digest == expected_digest, (
            f"{filename} does not match vocab/FROZEN.sha256 -- if this is intentional, see "
            "vocab/README.md (DECISIONS entry + human approval required, D-012)."
        )


def test_mutated_vocab_file_would_be_detected(tmp_path: Path) -> None:
    """Sanity-check the check itself: a modified copy must fail against the recorded hash."""
    original = VOCAB_DIR / "generic_phrases.yaml"
    mutated = tmp_path / "generic_phrases.yaml"
    mutated.write_bytes(original.read_bytes() + b"\ntampered: true\n")

    expected_digest = _parse_frozen_file(FROZEN_FILE)["generic_phrases.yaml"]

    assert _sha256_of(mutated) != expected_digest
