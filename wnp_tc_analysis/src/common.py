from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import yaml


WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent
CONFIG_PATH = WORK / "config" / "analysis.yml"
CURRENT_PATH = WORK / "CURRENT"
RELEASE_TAG = os.environ.get(
    "WNP_TC_RELEASE_TAG",
    CURRENT_PATH.read_text(encoding="utf-8").strip(),
).strip()
if not re.fullmatch(r"v\d+_\d+", RELEASE_TAG):
    raise RuntimeError(
        f"Invalid WNP TC analysis release tag {RELEASE_TAG!r}; "
        f"check {CURRENT_PATH} or WNP_TC_RELEASE_TAG."
    )
RELEASE = WORK / "releases" / RELEASE_TAG
MANUSCRIPT_MD = RELEASE / "manuscript" / "MD"
MANUSCRIPT_DOCX = RELEASE / "manuscript" / "DOCX"
MAIN_FIGURES = RELEASE / "figures" / "Main"
SUPPLEMENTARY_FIGURES = RELEASE / "figures" / "Supplementary"
REPORTS = RELEASE / "reports"
RELEASE_QA = RELEASE / "qa"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def ensure_output_dirs() -> None:
    cfg = load_config()
    for value in cfg["outputs"].values():
        Path(value).mkdir(parents=True, exist_ok=True)
    for name in [
        "01_landfall_latitude",
        "02_cutpoint_sensitivity",
        "03_common_storms",
        "04_track_density_sensitivity",
        "05_climate_mode_adjustment",
        "06_landfall_grouping",
        "07_wnpsh_dynamic_metric",
        "08_cluster_validation",
        "09_count_model_and_fdr",
    ]:
        (WORK / "analysis" / name).mkdir(parents=True, exist_ok=True)
    for path in [MANUSCRIPT_MD, MANUSCRIPT_DOCX, MAIN_FIGURES, SUPPLEMENTARY_FIGURES, REPORTS, RELEASE_QA]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def environment_record() -> dict:
    try:
        import importlib.metadata as md
        packages = {}
        for name in ["numpy", "pandas", "scipy", "statsmodels", "xarray", "geopandas", "shapely", "matplotlib", "scikit-learn", "PyYAML"]:
            try:
                packages[name] = md.version(name)
            except md.PackageNotFoundError:
                packages[name] = "not-installed"
    except Exception:
        packages = {}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "random_seed": load_config()["random_seed"],
    }
