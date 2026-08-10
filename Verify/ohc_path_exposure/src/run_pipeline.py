"""Review-gated orchestration for the isolated OHC exposure pipeline."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    PACKAGE_ROOT,
    load_config,
    project_path,
    require_execute,
    resolve_config_path,
    sha256_file,
    validate_code_review_gate,
    validate_seam_review_gate,
)


PREPARE_STAGES = [
    "prepare_ohc_region.py",
    "extract_agency_tracks.py",
    "match_tracks_to_ohc.py",
    "audit_product_seam.py",
]

ANALYZE_STAGES = [
    "decompose_ohc_exposure.py",
]


def write_manifest(config: dict) -> None:
    results = PACKAGE_ROOT / "results"
    manifest_path = project_path(config["pipeline"]["manifest_csv"])
    files = sorted(path for path in results.rglob("*") if path.is_file())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in files:
            writer.writerow({
                "path": str(path.relative_to(project_path("."))),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["prepare", "analyze"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require_execute(args.execute)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    validate_code_review_gate(config_path, config)
    if args.phase == "analyze":
        validate_seam_review_gate(config_path, config)

    stages = PREPARE_STAGES if args.phase == "prepare" else ANALYZE_STAGES
    for stage in stages:
        command = [sys.executable, str(PACKAGE_ROOT / "src" / stage), "--config", str(config_path), "--execute"]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True, cwd=project_path("."))
    write_manifest(config)


if __name__ == "__main__":
    main()
