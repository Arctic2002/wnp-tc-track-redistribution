"""为稳健性矩阵的每个配置计算有符号投影 S_c。

动机（外部审阅 P0-06）：正文称"106 种配置方向一致"，但 Table S1 只报告无符号的
TV 和 p 值，且 106 行中仅 18 行有 spatial correlation。读者无法从表本身验证该结论。

定义：S_c = d_c·d_p / (d_p·d_p)，d_p 为主配置（PRIMARY、track_point、2.5°、
1966–1995 vs 1996–2025）的两期差异场。S_c>0 表示该配置的差异场在主对比方向上为正投影，
即"方向一致"；S_c 同时给出相对于主配置的幅度。另报告空间相关 r 以区分"方向一致"与
"形态相似"——两者不是一回事，前者只要求投影为正，后者要求整体形态接近。

数据来源：`Verify/supplemental_audit_new_findings/new_outputs/wnp_tc_redistribution_pattern.npz`
存有四套记录 × 两种权重在 2.5° 网格上的逐年相对路径场（60 年 × 512 格）。由这些年度场
可直接重算任意时期端点组合的 d_c，无需重跑主管线。block length 只影响置换 p 值，不影响
d_c，因此同一 catalog／weighting／时期的不同 block 行共享同一 S_c。

网格不同的配置（1°、5°）与强度阈值配置（typhoon-only）的年度场不在该存档内。它们的
d_c 定义在不同网格或不同样本上，与 d_p 不同构，投影无法直接定义。本脚本对这些行
输出空值并标记原因，不做regrid近似——把不同网格的场强行插值到主网格会引入与结论无关的
插值假设。这些行改由"该配置自身网格上的两期差异符号"单独说明。

自检：对每一行用重算的 d_c 复算 TV，与存档 TV 比对；不一致即中止，避免在错位的
年度场上给出投影。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Verify/supplemental_audit_new_findings/new_outputs"
OUT = Path(__file__).resolve().parents[1] / "results"
MATRIX = SRC / "wnp_tc_robustness_matrix.csv"
PATTERN = SRC / "wnp_tc_redistribution_pattern.npz"

TOL = 1e-6

# `drop_*` 配置在保持时期端点不变的前提下剔除整个年代。端点列不体现这一点，
# 只能由 analysis 名解析；剔除后必须与存档的 n_early/n_late 对得上，否则中止。
DROPPED_YEARS = {
    "drop_1970s": range(1970, 1980),
    "drop_1980s": range(1980, 1990),
    "drop_1990s": range(1990, 2000),
    "drop_2000s": range(2000, 2010),
    "drop_2010s": range(2010, 2020),
    "drop_2020_2025": range(2020, 2026),
}


def field_key(catalog: str, weighting: str) -> str:
    return f"{catalog}_{weighting}_annual_fields"


def contrast(fields, years, es, ee, ls, le, dropped=()):
    keep = ~np.isin(years, list(dropped))
    early = (years >= es) & (years <= ee) & keep
    late = (years >= ls) & (years <= le) & keep
    if not early.any() or not late.any():
        raise ValueError(f"empty period: {es}-{ee} / {ls}-{le}")
    return fields[late].mean(0) - fields[early].mean(0), int(early.sum()), int(late.sum())


def main() -> None:
    z = np.load(PATTERN, allow_pickle=True)
    years = z["years"]
    rows = list(csv.DictReader(MATRIX.open(encoding="utf-8")))

    d_primary, _, _ = contrast(z[field_key("PRIMARY", "track_point")], years,
                               1966, 1995, 1996, 2025)
    denom = float(d_primary @ d_primary)

    out_rows = []
    checked = 0
    skipped: dict[str, int] = {}
    for r in rows:
        rec = dict(r)
        grid = r["grid_deg"]
        cat = r["catalog"]
        key = field_key(cat, r["weighting"])
        if grid != "2.5" or key not in z.files:
            reason = ("grid_differs_from_primary" if grid != "2.5"
                      else "sample_differs_from_primary")
            rec["signed_projection"] = ""
            rec["spatial_r_vs_primary"] = ""
            rec["projection_status"] = reason
            skipped[reason] = skipped.get(reason, 0) + 1
            out_rows.append(rec)
            continue

        d_c, n_e, n_l = contrast(z[key], years,
                                 int(r["early_start"]), int(r["early_end"]),
                                 int(r["late_start"]), int(r["late_end"]),
                                 DROPPED_YEARS.get(r["analysis"], ()))
        if (n_e, n_l) != (int(r["n_early"]), int(r["n_late"])):
            raise SystemExit(
                f"样本年数不符，中止：{r['analysis']}/{cat}/{r['weighting']} "
                f"存档=({r['n_early']},{r['n_late']}) 复算=({n_e},{n_l})")
        tv = 0.5 * float(np.abs(d_c).sum())
        if abs(tv - float(r["tv"])) > TOL:
            raise SystemExit(
                f"TV 复算不符，中止：{r['analysis']}/{cat}/{r['weighting']} "
                f"存档={r['tv']} 复算={tv}")
        checked += 1
        rec["signed_projection"] = f"{float(d_c @ d_primary) / denom:.4f}"
        rec["spatial_r_vs_primary"] = f"{float(np.corrcoef(d_c, d_primary)[0, 1]):.4f}"
        rec["projection_status"] = "computed"
        out_rows.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys())
    with (OUT / "robustness_matrix_signed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    computed = [r for r in out_rows if r["projection_status"] == "computed"]
    pos = [r for r in computed if float(r["signed_projection"]) > 0]
    sig = [r for r in out_rows if r["block_permutation_p"]
           and float(r["block_permutation_p"]) < 0.05]
    summary = {
        "n_configurations": len(out_rows),
        "n_projection_computed": len(computed),
        "n_projection_positive": len(pos),
        "n_projection_negative": len(computed) - len(pos),
        "n_p_lt_0_05": len(sig),
        "tv_reproduced": checked,
        "skipped": skipped,
        "min_signed_projection": min(float(r["signed_projection"]) for r in computed),
        "max_signed_projection": max(float(r["signed_projection"]) for r in computed),
        "min_spatial_r": min(float(r["spatial_r_vs_primary"]) for r in computed),
    }
    (OUT / "signed_projection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
