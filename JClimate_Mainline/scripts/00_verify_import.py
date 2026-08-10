from __future__ import annotations

import csv
import ast
import hashlib
from pathlib import Path


WORK = Path(__file__).resolve().parents[1]
SOURCE = WORK.parent / "Revision" / "v2"
MANIFEST = WORK / "manifests" / "subproject_import_manifest.csv"
REPORT = WORK / "docs" / "SUBPROJECT_VALIDATION_20260721.md"

EXPECTED_LOCAL_CHANGES = {
    "README.md",
    "VERSION.md",
    "config/analysis.yml",
    "docs/SUBPROJECT_IMPORT_20260721.md",
    "docs/SUBPROJECT_VALIDATION_20260721.md",
    "docs/CHANGELOG.md",
    "manifests/subproject_import_manifest.csv",
    "releases/v1_0/README.md",
    "releases/v1_0/reports/delivery_validation_v1_0.md",
    "releases/v1_0/reports/version_manifest_v1_0.csv",
    "scripts/00_verify_import.py",
    "scripts/common.py",
    "scripts/39_jcli_formal_figures.py",
    "scripts/41_build_docx.py",
    "scripts/42_build_cn_reference_layout.py",
    "scripts/44_audit_package.py",
    "src/audit.py",
    "src/build_release_docx.py",
    "src/common.py",
    "src/compare_rendered_pages.py",
    "src/finalize_registry_and_reports.py",
    "src/validate_release_delivery.py",
    "tests/conftest.py",
}

KEY_FILES = [
    "releases/v1_0/manuscript/MD/Manuscript_CN_v1_0.md",
    "releases/v1_0/manuscript/MD/Manuscript_EN_v1_0.md",
    "releases/v1_0/manuscript/MD/Supplementary_v1_0.md",
    "releases/v1_0/manuscript/DOCX/Manuscript_CN_v1_0_中文版式.docx",
    "releases/v1_0/manuscript/DOCX/Manuscript_EN_v1_0.docx",
    "releases/v1_0/manuscript/DOCX/Supplementary_v1_0.docx",
    "releases/v1_0/reports/results_registry.csv",
    "releases/v1_0/reports/final_agent_report.md",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_for(relative: str) -> Path:
    """Map independent subproject v1.0 names back to Revision v2.1 provenance."""
    mapped = relative
    if mapped.startswith("releases/v1_0/"):
        mapped = "releases/v2_1/" + mapped.removeprefix("releases/v1_0/")
        mapped = mapped.replace("_v1_0", "_v2_1")
    return SOURCE / mapped


def main() -> None:
    syntax_errors = []
    for folder in (WORK / "src", WORK / "scripts", WORK / "tests"):
        for python_file in folder.rglob("*.py"):
            try:
                ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))
            except (SyntaxError, UnicodeDecodeError) as exc:
                syntax_errors.append(f"{python_file.relative_to(WORK).as_posix()}: {exc}")

    rows = []
    for path in sorted(WORK.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(WORK).as_posix()
        source = source_for(relative)
        current_hash = digest(path)
        source_hash = digest(source) if source.exists() else ""
        if relative in EXPECTED_LOCAL_CHANGES:
            classification = "subproject-local"
        elif source.exists() and current_hash == source_hash:
            classification = "copied-unchanged"
        elif source.exists():
            classification = "unexpected-difference"
        else:
            classification = "subproject-only"
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": current_hash,
                "source_exists": source.exists(),
                "source_sha256": source_hash,
                "matches_source": bool(source.exists() and current_hash == source_hash),
                "classification": classification,
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unexpected = [row for row in rows if row["classification"] == "unexpected-difference"]
    key_lines = []
    for relative in KEY_FILES:
        current = WORK / relative
        source = source_for(relative)
        ok = current.exists() and source.exists() and digest(current) == digest(source)
        key_lines.append(f"- {'PASS' if ok else 'FAIL'}：`{relative}`")

    copied = sum(row["classification"] == "copied-unchanged" for row in rows)
    local = sum(row["classification"] in {"subproject-local", "subproject-only"} for row in rows)
    report = [
        "# JClimate Mainline 子项目核验",
        "",
        "## 结论",
        "",
        f"- 盘点文件：{len(rows)}个。",
        f"- 与 `Revision/v2` 哈希一致：{copied}个。",
        f"- 子项目说明或路径适配文件：{local}个。",
        f"- 非预期差异：{len(unexpected)}个。",
        f"- Python语法错误：{len(syntax_errors)}个。",
        "- 本次导入采用复制方式，来源版本未移动或覆盖。",
        "",
        "## 关键交付文件哈希核对",
        "",
        *key_lines,
        "",
        "## 非预期差异",
        "",
    ]
    if unexpected:
        report.extend(f"- `{row['relative_path']}`" for row in unexpected)
    else:
        report.append("- 无。")
    report.extend(["", "## Python语法检查", ""])
    if syntax_errors:
        report.extend(f"- `{item}`" for item in syntax_errors)
    else:
        report.append("- PASS：`src/`、`scripts/`和`tests/`中的Python文件均可解析。")
    report.extend(
        [
            "",
            "## 说明",
            "",
            "允许变化的文件仅包括子项目README、版本与导入记录、工作路径配置，以及为适配新目录层级而修改的路径解析代码。科研数据、结果、稿件和图件不在允许变化清单内。",
            "",
        ]
    )
    REPORT.write_text("\n".join(report), encoding="utf-8")
    if unexpected or syntax_errors or any("FAIL" in line for line in key_lines):
        raise SystemExit("subproject import validation failed")
    print(f"validated {len(rows)} files; copied unchanged={copied}; local={local}")


if __name__ == "__main__":
    main()
