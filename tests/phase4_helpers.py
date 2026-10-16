"""Public synthetic-suite fixture; no private manifest access."""
from pathlib import Path
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def validation_policy():
    return yaml.safe_load((ROOT/"config/validation.yaml").read_text(encoding="utf-8"))
