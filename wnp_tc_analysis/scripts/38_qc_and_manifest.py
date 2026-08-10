from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, RESULTS, WORK, ensure_dirs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    ensure_dirs()
    checks = []

    annual = pd.read_csv(DATA / "wnp_tc_redistribution_index_annual.csv")
    checks.append(("annual_index_rows_480", len(annual) == 480, len(annual)))
    checks.append(("annual_index_no_nan", annual[["index_full", "index_oos"]].notna().all().all(), int(annual[["index_full", "index_oos"]].isna().sum().sum())))
    checks.append(("annual_index_unique_keys", not annual.duplicated(["year", "agency", "weighting"]).any(), int(annual.duplicated(["year", "agency", "weighting"]).sum())))

    decomp = pd.read_csv(DATA / "wnp_tc_genesis_propagation_summary.csv")
    checks.append(("decomposition_closure", float(decomp.closure_max_abs.max()) < 1e-12, float(decomp.closure_max_abs.max())))
    checks.append(("decomposition_fraction_sum", np.allclose(decomp.genesis_projection_fraction + decomp.propagation_projection_fraction, 1), float(np.max(np.abs(decomp.genesis_projection_fraction + decomp.propagation_projection_fraction - 1)))))

    land = pd.read_csv(DATA / "wnp_tc_landfall_unique_events.csv")
    dup = land.duplicated(["agency", "assignment_rule", "sid"]).sum()
    checks.append(("unique_landfall_one_row_per_storm", dup == 0, int(dup)))

    circ = pd.read_csv(DATA / "wnp_tc_circulation_regression_summary.csv")
    checks.append(("circulation_six_field_tests", len(circ) == 6, len(circ)))
    checks.append(("circulation_p_range", circ.global_block_permutation_p.between(0, 1).all(), ""))

    qc = pd.DataFrame(checks, columns=["check", "passed", "value"])
    qc.to_csv(RESULTS / "qc_summary.csv", index=False)

    source_manifest = WORK / "manifests" / "source_manifest.csv"
    if not source_manifest.exists():
        raise SystemExit(f"missing frozen source manifest: {source_manifest}")
    expected = pd.read_csv(source_manifest)
    integrity = []
    for row in expected.itertuples(index=False):
        source = WORK / row.relative_path
        actual = digest(source) if source.exists() else ""
        integrity.append({
            "file": row.relative_path,
            "source_exists": source.exists(),
            "expected_sha256": row.sha256,
            "actual_sha256": actual,
            "unchanged": source.exists() and actual == row.sha256,
        })
    pd.DataFrame(integrity).to_csv(RESULTS / "source_integrity_check.csv", index=False)

    manifest_path = WORK / "manifests" / "file_manifest.csv"
    manifest = []
    for path in sorted(WORK.rglob("*")):
        if path.is_file() and path != manifest_path:
            manifest.append({"relative_path": str(path.relative_to(WORK)), "bytes": path.stat().st_size, "sha256": digest(path)})
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    if not qc.passed.all():
        raise SystemExit(qc.to_string(index=False))
    if not pd.DataFrame(integrity).unchanged.all():
        raise SystemExit("one or more frozen source files changed")
    print(qc.to_string(index=False))
    print(f"source files unchanged: {len(integrity)}")
    print(f"manifest files: {len(manifest)}")


if __name__ == "__main__":
    run()
