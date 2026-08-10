from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .common import PROJECT, REPORTS, WORK, ensure_output_dirs, environment_record, load_config, sha256, write_json


REGISTRY_COLUMNS = [
    "manuscript_section", "claim_id", "metric", "dataset", "period",
    "spatial_domain", "method", "estimate", "uncertainty", "p_value",
    "q_value", "source_script", "source_output", "status", "notes",
]


def file_row(path: Path, role: str) -> dict:
    return {
        "role": role,
        "path": str(path.relative_to(PROJECT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
        "modified": path.stat().st_mtime if path.exists() else "",
        "sha256": sha256(path) if path.exists() and path.is_file() and path.stat().st_size < 200_000_000 else "see external_upstream_manifest.csv",
    }


def netcdf_row(path: Path) -> dict:
    with xr.open_dataset(path) as ds:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_res = float(abs(np.diff(ds[lat_name].values[:2])[0])) if ds.sizes.get(lat_name, 0) > 1 else np.nan
        lon_res = float(abs(np.diff(ds[lon_name].values[:2])[0])) if ds.sizes.get(lon_name, 0) > 1 else np.nan
        return {
            "path": str(path.relative_to(PROJECT)),
            "variables": ";".join(ds.data_vars),
            "dimensions": ";".join(f"{k}={v}" for k, v in ds.sizes.items()),
            "time_start": str(ds.time.min().values)[:10] if "time" in ds else "",
            "time_end": str(ds.time.max().values)[:10] if "time" in ds else "",
            "lat_resolution_deg": lat_res,
            "lon_resolution_deg": lon_res,
        }


def ibtracs_completeness(path: Path) -> pd.DataFrame:
    columns = ["SID", "SEASON", "ISO_TIME", "TRACK_TYPE", "IFLAG", "USA_WIND", "TOKYO_WIND", "CMA_WIND"]
    frame = pd.read_csv(path, usecols=columns, skiprows=[1], low_memory=False)
    frame["SEASON"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    rows = []
    for year in range(2015, 2026):
        part = frame.loc[frame.SEASON.eq(year)]
        row = {"season": year, "n_sid": part.SID.nunique(), "n_rows": len(part), "last_time": part.ISO_TIME.max()}
        for agency in ["USA", "TOKYO", "CMA"]:
            valid = pd.to_numeric(part[f"{agency}_WIND"], errors="coerce").notna()
            row[f"n_sid_{agency.lower()}_wind"] = part.loc[valid, "SID"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def manuscript_mentions() -> pd.DataFrame:
    rows = []
    keyword = re.compile(r"(Sen|TVD|总变差|置换|p\s*=|q\s*=|斜率|相关|百分点|LMI|登陆|PDO|ONI|FDR|588)", re.I)
    number = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:%|—\d+(?:\.\d+)?|–\d+(?:\.\d+)?)?")
    for manuscript in [WORK / "MD" / "Manuscript_CN.md", WORK / "MD" / "Manuscript_EN.md", WORK / "MD" / "Supplementary.md"]:
        for line_no, line in enumerate(manuscript.read_text(encoding="utf-8").splitlines(), 1):
            if keyword.search(line) and number.search(line):
                rows.append({"manuscript": manuscript.name, "line": line_no, "numbers": ";".join(number.findall(line)), "text": line.strip()})
    return pd.DataFrame(rows)


def initial_registry() -> pd.DataFrame:
    rows: list[dict] = []
    trend_path = WORK / "data" / "upstream_revision" / "p2_multiagency_trends.csv"
    trends = pd.read_csv(trend_path)
    selected = trends.loc[(trends.end == 2025) & trends.analysis.isin(["frequency", "lmi_full_catalog", "lmi_common_storms"])]
    for r in selected.itertuples():
        rows.append(dict(
            manuscript_section="Results 3.1/3.2", claim_id=f"trend_{r.analysis}_{r.agency}", metric=r.variable,
            dataset=r.agency.replace("TOKYO", "JMA"), period=f"{r.start}-{r.end}", spatial_domain="WNP",
            method="Theil-Sen; Hamed-Rao MK; BH-FDR within named family", estimate=r.sen_slope_per_decade,
            uncertainty=f"95% CI [{r.sen_ci_lo_per_decade}, {r.sen_ci_hi_per_decade}]", p_value=r.mk_p_raw,
            q_value=r.mk_p_fdr_bh, source_script="scripts/upstream_revision/29_multiagency_sensitivity.py",
            source_output=str(trend_path.relative_to(WORK)), status="verified_against_existing_output", notes=f"FDR family: {r.fdr_family}",
        ))
    red_path = WORK / "data" / "upstream_revision" / "p2_multiagency_redistribution.csv"
    for r in pd.read_csv(red_path).itertuples():
        rows.append(dict(
            manuscript_section="Results 3.2/3.4", claim_id=f"redistribution_{r.analysis}_{r.agency}", metric=r.analysis,
            dataset=r.agency.replace("TOKYO", "JMA"), period=f"{r.early_start}-{r.early_end} vs {r.late_start}-{r.late_end}",
            spatial_domain="100-180E, 0-40N" if r.analysis == "path_density" else "WNP landfall coasts",
            method=f"annual composition; TVD; {r.block_years}-yr block permutation ({r.n_permutations})",
            estimate=r.total_variation, uncertainty="", p_value=r.block_permutation_p, q_value="",
            source_script="scripts/upstream_revision/29_multiagency_sensitivity.py", source_output=str(red_path.relative_to(WORK)),
            status="verified_against_existing_output", notes="No FDR applied to pre-specified field-level permutation test",
        ))
    agreement_path = WORK / "data" / "upstream_revision" / "p2_multiagency_agreement.csv"
    for r in pd.read_csv(agreement_path).itertuples():
        if r.metric in {"path_change_map", "annual_lmi_lat_common"}:
            rows.append(dict(
                manuscript_section="Results 3.2", claim_id=f"agreement_{r.metric}_{r.agency_left}_{r.agency_right}", metric=r.metric,
                dataset=f"{r.agency_left.replace('TOKYO','JMA')}-{r.agency_right.replace('TOKYO','JMA')}",
                period="1966-2025" if r.metric == "path_change_map" else "1982-2025", spatial_domain="WNP",
                method="Pearson spatial/annual correlation", estimate=r.correlation, uncertainty="", p_value="", q_value="",
                source_script="scripts/upstream_revision/29_multiagency_sensitivity.py", source_output=str(agreement_path.relative_to(WORK)),
                status="verified_against_existing_output", notes="Agency records share observational basis and are not independent experiments",
            ))
    unique_path = WORK / "data" / "wnp_tc_landfall_unique_summary.csv"
    for r in pd.read_csv(unique_path).itertuples():
        rows.append(dict(
            manuscript_section="Results 3.6", claim_id=f"unique_landfall_{r.agency}_{r.assignment_rule}", metric="north_share_change_percentage_points",
            dataset=r.agency.replace("TOKYO", "JMA"), period="1966-1995 vs 1996-2025", spatial_domain="named WNP coast groups",
            method=f"unique-storm assignment={r.assignment_rule}; 3-yr block permutation", estimate=r.north_share_change_percentage_points,
            uncertainty="", p_value=r.north_share_block_p, q_value="", source_script="scripts/36_landfall_unique_assignment.py",
            source_output=str(unique_path.relative_to(WORK)), status="verified_against_existing_output",
            notes="Raster-densified crossing input; exact coastline-intersection latitude analysis still required",
        ))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def run() -> None:
    ensure_output_dirs()
    cfg = load_config()
    tables = Path(cfg["outputs"]["tables"])
    inventory_paths = [
        (PROJECT / "data/raw/IBTrACS.WP.v04r01.csv", "best-track original"),
        (PROJECT / "data/raw/ibtracs_wp.csv", "best-track pipeline copy"),
        (PROJECT / "data/raw/indices/oni.csv", "ENSO index"),
        (PROJECT / "data/raw/indices/pdo.csv", "PDO index"),
        (PROJECT / "data/raw/GSHHG/GSHHS_shp/h/GSHHS_h_L1.shp", "high-resolution coastline"),
        (PROJECT / "config/config.yaml", "project configuration"),
        (PROJECT / "config/environment.yml", "environment definition"),
        (WORK / "MD/Manuscript_CN.md", "current Chinese manuscript"),
        (WORK / "MD/Manuscript_EN.md", "current English manuscript"),
        (WORK / "DOCX/Manuscript_CN.docx", "current Chinese DOCX"),
        (WORK / "sources/CODEX_论文系统修订任务书.md", "revision specification"),
    ]
    pd.DataFrame([file_row(p, role) for p, role in inventory_paths]).to_csv(tables / "project_file_inventory.csv", index=False)
    nc_paths = [PROJECT / "data/interim" / n for n in ["era5_wnp_single.nc", "era5_wnp_plev.nc", "era5_wnp_dynamic_plev.nc", "steering.nc"]]
    pd.DataFrame([netcdf_row(p) for p in nc_paths]).to_csv(tables / "era5_inventory.csv", index=False)
    completeness = ibtracs_completeness(PROJECT / "data/raw/IBTrACS.WP.v04r01.csv")
    completeness.to_csv(tables / "ibtracs_recent_year_completeness.csv", index=False)
    manuscript_mentions().to_csv(tables / "manuscript_numeric_mentions.csv", index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    initial_registry().to_csv(REPORTS / "results_registry.csv", index=False)
    write_json(WORK / "outputs/logs/environment.json", environment_record())

    script_counts = {name: len(list(path.rglob("*.py"))) for name, path in {
        "core": PROJECT / "core", "paper1": PROJECT / "paper1_thermo", "paper2": PROJECT / "paper2_dynamic", "revision_v2": WORK / "scripts"}.items()}
    figure_counts = {"main": len(list((WORK / "Figures/Main").glob("*.*"))), "supplementary": len(list((WORK / "Figures/Supplementary").glob("*.*")))}
    y2025 = completeness.loc[completeness.season.eq(2025)].iloc[0]
    report = f"""# 00 项目审计

审计日期：2026-07-21  
执行规范：`sources/CODEX_论文系统修订任务书.md`  
工作副本：`wnp_tc_analysis`；来源版本`Revision/v2`和项目根目录原始数据均未覆盖。

## 1. 当前文稿与输入边界

- 当前中文、英文及补充材料基线位于`MD/`，对应Word稿位于`DOCX/`。
- 任务书指定的导师审阅稿`4b0468af-157c-4f90-8eec-69a816a3561a.docx`未在项目或`D:/Download`找到。当前审计使用已完成历次导师意见整合的v1稿作为基线；未知批注不得推测写回。
- 目标期刊尚未最终确定。现有稿按早期候选格式组织，但仍生成`journal_style_pending.md`。

## 2. 数据清点

- 最佳路径：IBTrACS WP v04r01，原文件`data/raw/IBTrACS.WP.v04r01.csv`；本地文件时间为2026-06-15。流水线副本及元数据位于`data/raw/ibtracs_wp.csv`和`ibtracs_wp.metadata.json`。
- 机构映射：USA使用`USA_LAT/LON/WIND/STATUS`，JMA在IBTrACS中对应`TOKYO_LAT/LON/WIND/GRADE`，CMA使用`CMA_LAT/LON/WIND/CAT`；原始/机构报告由`IFLAG`筛选。
- ERA5四个核心文件的变量、维度、时段和分辨率见`outputs/tables/era5_inventory.csv`，月资料覆盖1940-01至2025-12。
- ENSO/PDO：`data/raw/indices/oni.csv`和`pdo.csv`；来源说明需在方法和引用审计中继续核对。
- 海岸线：GSHHG高分辨率L1陆地多边形。既有登陆算法采用0.02°掩膜和0.01°线段加密，新P0分析改用轨迹线段与海岸线的几何交点。
- 环境定义：`config/environment.yml`；实际运行环境见`outputs/logs/environment.json`。

## 3. 2025年完整性

- 本地IBTrACS文件含2025年{int(y2025.n_sid)}个SID、{int(y2025.n_rows)}行记录，最后时间为{pd.Timestamp(y2025.last_time)}；USA/JMA/CMA含风速记录的SID分别为{int(y2025.n_sid_usa_wind)}、{int(y2025.n_sid_tokyo_wind)}、{int(y2025.n_sid_cma_wind)}。
- ERA5月资料覆盖到2025-12。可确认本地文件年度覆盖完整，但仅凭仓库无法证明IBTrACS对2025年的业务最终定版状态。因此保留1966—2025主分析，同时执行剔除2025敏感性。

## 4. 代码、统计与图件

- Python脚本数量：Core {script_counts['core']}、Paper I {script_counts['paper1']}、Paper II {script_counts['paper2']}、Revision v2既有脚本 {script_counts['revision_v2']}。
- 当前正式主图文件{figure_counts['main']}个、补充图文件{figure_counts['supplementary']}个；它们是审计基线，不等于任务书要求的最终主图结构。
- 已有随机种子主要为202406；新分析统一由`config/analysis.yml`控制。
- `results_registry.csv`已登记频数、LMI、路径场、机构一致性和唯一风暴登陆结果；后续复算后更新状态。

## 5. 初步一致性结论

- 文稿中的三机构频数斜率、路径场TVD/p值、路径空间相关和唯一风暴登陆份额变化均能在现有CSV中定位。
- 既有登陆事件由栅格加密算法生成，尚不能满足任务书的高分辨率海岸线几何交点要求；直接登陆纬度当前未验证。
- 当前主稿已把登陆结论限制为海岸构成/方向性变化，没有把唯一风暴结果写成显著的普遍北移；标题仍需由P0直接纬度分析决定。
- 固定588 dagpm不应解释为动力增强。当前稿件主要使用涡动位势高度和引导气流背景，但旧稿及旧图仍需全文检索清理。

## 6. 审计产物

- `outputs/tables/project_file_inventory.csv`
- `outputs/tables/era5_inventory.csv`
- `outputs/tables/ibtracs_recent_year_completeness.csv`
- `outputs/tables/manuscript_numeric_mentions.csv`
- `results_registry.csv`
- `outputs/logs/environment.json`

## 7. 下一步

先执行P0直接登陆纬度和三机构核心结果复算，再重构主图与文稿。P1分析不得通过寻找最显著口径替换预设主分析。
"""
    audit_dir = WORK / "docs" / "system_revision"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "00_project_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
