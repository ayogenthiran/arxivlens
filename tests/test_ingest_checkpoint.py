import json
from pathlib import Path

import pytest

from main import _read_checkpoint


def test_missing_checkpoint_starts_at_zero(tmp_path: Path):
    assert _read_checkpoint(str(tmp_path / "missing.json")) == 0


def test_checkpoint_returns_processed_docs(tmp_path: Path):
    path = tmp_path / "ingest_checkpoint.json"
    path.write_text(json.dumps({"processed_docs": 12}), encoding="utf-8")
    assert _read_checkpoint(str(path)) == 12


def test_corrupt_checkpoint_is_not_treated_as_a_fresh_start(tmp_path: Path):
    path = tmp_path / "ingest_checkpoint.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _read_checkpoint(str(path))
