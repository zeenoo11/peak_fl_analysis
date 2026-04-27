"""Load v10 train/cold household split from external yaml.

The split was originally produced by Peak_Analysis/experiments/federated/
v10_0425_cold_split.py (KMeans-based stratified 50/50). v11 inherits it for
direct comparability.
"""

from __future__ import annotations

from pathlib import Path

import yaml

V10_YAML = (
    Path(__file__).resolve().parents[3]
    / "Peak_Analysis"
    / "configs"
    / "v10_households.yaml"
)


def load_v10_split(yaml_path: Path = V10_YAML) -> dict[str, list[str]]:
    """Return {'train': [...50 apts...], 'cold': [...50 apts...]}."""
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"v10 split yaml missing: {yaml_path}. "
            "v11 depends on the v10 train/cold split for comparability."
        )
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)
    return {"train": list(raw["train"]), "cold": list(raw["cold"])}
