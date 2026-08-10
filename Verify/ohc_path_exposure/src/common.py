"""Shared guards and path handling for the isolated OHC exposure audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PACKAGE_ROOT / "config.json"


def coordinate_grids_equivalent(
    current: np.ndarray,
    reference: np.ndarray,
    atol_degrees: float,
) -> bool:
    """Treat sub-grid float32 coordinate quantization as equivalent geometry."""
    current = np.asarray(current, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    if current.shape != reference.shape:
        return False
    return bool(
        np.allclose(
            current,
            reference,
            rtol=0.0,
            atol=float(atol_degrees),
            equal_nan=True,
        )
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_bundle_files(config_path: Path) -> list[Path]:
    files = [
        Path(config_path).resolve(),
        PACKAGE_ROOT / "README.md",
        PACKAGE_ROOT / "requirements.txt",
        PACKAGE_ROOT / "docs" / "METHOD_SPEC.md",
        PACKAGE_ROOT / "docs" / "OUTPUT_SCHEMA.md",
        PACKAGE_ROOT / "docs" / "THIRD_PARTY_REVIEW.md",
    ]
    files.extend(sorted((PACKAGE_ROOT / "src").glob("*.py")))
    files.extend(sorted((PACKAGE_ROOT / "tests").glob("test_*.py")))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"review bundle is incomplete: {missing}")
    return sorted(set(path.resolve() for path in files), key=lambda path: str(path).lower())


def review_bundle_sha256(config_path: Path) -> str:
    digest = hashlib.sha256()
    for path in review_bundle_files(config_path):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema_version")
    return config


def resolve_config_path(path: Path | None = None) -> Path:
    return Path(path or DEFAULT_CONFIG).resolve()


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"path escapes project root: {resolved}")
    return resolved


def require_execute(enabled: bool) -> None:
    if not enabled:
        raise SystemExit(
            "execution guard: inspect the code first, then rerun with --execute"
        )


def validate_code_review_gate(config_path: Path, config: dict) -> dict:
    approval_path = project_path(config["pipeline"]["approval_file"])
    if not approval_path.is_file():
        raise PermissionError(
            f"third-party review approval is absent: {approval_path}; do not copy the example unchanged"
        )
    with approval_path.open("r", encoding="utf-8") as handle:
        approval = json.load(handle)
    if approval.get("status") != "approved" or int(approval.get("open_blocking_items", -1)) != 0:
        raise PermissionError("third-party review has not approved execution")
    if approval.get("config_sha256") != sha256_file(config_path):
        raise PermissionError("approval config_sha256 does not match the current config.json")
    if approval.get("review_bundle_sha256") != review_bundle_sha256(config_path):
        raise PermissionError("approval review_bundle_sha256 does not match the reviewed code bundle")
    return approval


def validate_seam_review_gate(config_path: Path, config: dict) -> dict:
    review_path = project_path(config["pipeline"]["seam_review_file"])
    audit_path = project_path(config["seam_audit"]["output_json"])
    if not review_path.is_file():
        raise PermissionError(f"manual product-seam review is absent: {review_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"product-seam audit output is absent: {audit_path}")
    with review_path.open("r", encoding="utf-8") as handle:
        review = json.load(handle)
    if review.get("status") != "reviewed" or review.get("decision") not in {
        "accept",
        "accept_with_limitations",
    }:
        raise PermissionError("product seam has not received an acceptable manual decision")
    if review.get("config_sha256") != sha256_file(config_path):
        raise PermissionError("seam review config_sha256 does not match the current config.json")
    if review.get("seam_audit_sha256") != sha256_file(audit_path):
        raise PermissionError("seam review is not bound to the current seam-audit output")
    return review


def ensure_output_target(path: Path, *, overwrite: bool) -> None:
    resolved = Path(path).resolve()
    if resolved != PACKAGE_ROOT and PACKAGE_ROOT not in resolved.parents:
        raise ValueError(f"derived output must remain under {PACKAGE_ROOT}: {resolved}")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    temp_path = Path(temp_path).resolve()
    final_path = Path(final_path).resolve()
    if temp_path.parent != final_path.parent:
        raise ValueError("atomic replacement requires the same parent directory")
    temp_path.replace(final_path)


def write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    ensure_output_target(path, overwrite=overwrite)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    atomic_replace(temp, path)
