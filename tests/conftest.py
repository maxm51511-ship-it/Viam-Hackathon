import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from printbot.config import DEFAULTS, _deep_merge  # noqa: E402


@pytest.fixture
def settings():
    """Geometry-mode settings (shape heuristics)."""
    return _deep_merge(DEFAULTS, {"logging": {"event_log": None}, "classifier": {"mode": "geometry"}})


@pytest.fixture
def demo_settings():
    """Hackathon demo mode: red = failed, white = good."""
    return _deep_merge(DEFAULTS, {"logging": {"event_log": None}, "classifier": {"mode": "colour_demo"}})
