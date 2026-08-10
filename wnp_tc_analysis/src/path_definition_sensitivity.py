from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .common import PROJECT, WORK, load_config
from .cutpoint_sensitivity import field_test
from .core_crossagency_recheck import safe_corr

from core.utils import haversine
from paper2_dynamic.agency_data import AGENCIES, build_agency_catalog, read_ibtracs_agencies


YEARS = np.arange(1966, 2026)
EARLY_YEARS = (1966, 1995)
LATE_YEARS = (1996, 2025)


def edges(width):
    return np.arange(100, 180 + width, width), np.arange(0, 40 + width, width)


def point_fields(tracks, width, weighting):
    lon_edges, lat_edges = edges(width)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    rows = []
    for year in YEARS:
        annual = tracks.loc[tracks["season"].eq(year)]
        if weighting == "track_point":
            field = np.histogram2d(annual["lon"], annual["lat"], bins=[lon_edges, lat_edges])[0].T
        else:
            field = np.zeros(shape, float)
            for _, storm in annual.groupby("sid"):
                one = np.histogram2d(storm["lon"], storm["lat"], bins=[lon_edges, lat_edges])[0].T
                if one.sum() == 0:
                    continue
                if weighting == "storm_equal":
                    field += one / one.sum()
                elif weighting == "binary_occupancy":
                    field += one > 0
                else:
                    raise ValueError(weighting)
        if field.sum() == 0:
            raise ValueError(f"empty {weighting} field in {year}")
        rows.append((field / field.sum()).ravel())
    return np.asarray(rows)


def truncate_before_first_landfall(points, first_times):
    data = points.merge(first_times, on=["agency", "sid"], how="left")
    keep = data["first_landfall_time"].isna() | (data["iso_time"] <= data["first_landfall_time"])
    return data.loc[keep, points.columns]


def line_length_fields(tracks, width):
    lon_edges, lat_edges = edges(width)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    rows = []
    for year in YEARS:
        field = np.zeros(shape, float)
        annual = tracks.loc[tracks["season"].eq(year)]
        for _, storm in annual.groupby("sid"):
            storm = storm.sort_values("iso_time")
            lat = storm["lat"].to_numpy(float)
            lon = storm["lon"].to_numpy(float)
            time = pd.to_datetime(storm["iso_time"]).to_numpy()
            for i in range(len(storm) - 1):
                gap = (time[i + 1] - time[i]) / np.timedelta64(1, "h")
                if not (0 < gap <= 12):
                    continue
                distance = float(haversine(lat[i], lon[i], lat[i + 1], lon[i + 1]))
                if not np.isfinite(distance) or distance <= 0:
                    continue
                n = max(1, int(np.ceil(distance / 25.0)))
                f = (np.arange(n) + 0.5) / n
                sample_lat = lat[i] + f * (lat[i + 1] - lat[i])
                sample_lon = lon[i] + f * (lon[i + 1] - lon[i])
                weight = np.full(n, distance / n)
                one = np.histogram2d(sample_lon, sample_lat, bins=[lon_edges, lat_edges], weights=weight)[0].T
                field += one
        if field.sum() == 0:
            raise ValueError(f"empty line-length field in {year}")
        rows.append((field / field.sum()).ravel())
    return np.asarray(rows)


def test_fields(records, changes, agency, definition, width, fields, blocks, cfg, source_stage):
    early = np.flatnonzero((YEARS >= EARLY_YEARS[0]) & (YEARS <= EARLY_YEARS[1]))
    late = np.flatnonzero((YEARS >= LATE_YEARS[0]) & (YEARS <= LATE_YEARS[1]))
    selected = np.r_[early, late]
    if not np.array_equal(selected, np.arange(len(YEARS))):
        raise ValueError("path-definition periods must partition YEARS in chronological order")
    for block in blocks:
        tv, p = field_test(fields, len(early), block=block, nperm=cfg["n_permutations"], seed=cfg["random_seed"])
        change = fields[late].mean(axis=0) - fields[early].mean(axis=0)
        records.append({"agency": agency, "definition": definition, "source_stage": source_stage,
                        "grid_deg": width, "block_years": block, "total_variation": tv,
                        "block_permutation_p": p, "n_permutations": cfg["n_permutations"]})
        if block == 3:
            changes[(agency, definition, width)] = change


def run():
    cfg = load_config()
    out = WORK / "analysis" / "04_track_density_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    source = read_ibtracs_agencies(PROJECT / "data" / "raw" / "IBTrACS.WP.v04r01.csv", 1945, 2025)
    catalogs = {agency: build_agency_catalog(source, agency) for agency in AGENCIES}
    events = pd.read_csv(WORK / "analysis" / "01_landfall_latitude" / "landfall_events_exact.csv")
    events["time"] = pd.to_datetime(events["time"])
    first = events.sort_values("time").drop_duplicates(["agency", "sid"])[["agency", "sid", "time"]]
    first = first.rename(columns={"time": "first_landfall_time"})

    records, changes = [], {}
    for agency, catalog in catalogs.items():
        ts = catalog["ts_points"].loc[catalog["ts_points"]["season"].between(1966, 2025)].copy()
        ts["agency"] = agency
        full = catalog["tracks"].loc[catalog["tracks"]["season"].between(1966, 2025)].copy()
        full["agency"] = agency
        pre = truncate_before_first_landfall(ts, first)

        for definition, tracks, weighting, source_stage in [
            ("track_point", ts, "track_point", "native_TS_plus"),
            ("storm_equal", ts, "storm_equal", "native_TS_plus"),
            ("binary_occupancy", ts, "binary_occupancy", "native_TS_plus"),
            ("pre_landfall_track_point", pre, "track_point", "native_TS_plus_truncated"),
            ("full_life_track_point", full, "track_point", "all_original_positions_of_eligible_storms"),
        ]:
            fields = point_fields(tracks, 2.5, weighting)
            blocks = [2, 3, 4, 5] if definition in {"binary_occupancy", "pre_landfall_track_point"} else [3]
            test_fields(records, changes, agency, definition, 2.5, fields, blocks, cfg, source_stage)
        line = line_length_fields(ts, 2.5)
        test_fields(records, changes, agency, "line_length", 2.5, line, [3], cfg, "native_TS_plus")

        # Resolution checks for the newly introduced occupancy definition.
        for width in [1.0, 5.0]:
            binary = point_fields(ts, width, "binary_occupancy")
            test_fields(records, changes, agency, "binary_occupancy", width, binary, [3], cfg, "native_TS_plus")

    results = pd.DataFrame(records)
    results.to_csv(out / "path_definition_sensitivity.csv", index=False)
    corr_rows = []
    for definition in sorted(results["definition"].unique()):
        for width in sorted(results.loc[results["definition"].eq(definition), "grid_deg"].unique()):
            keys = [(agency, definition, width) for agency in AGENCIES]
            if not all(key in changes for key in keys):
                continue
            for left, right in combinations(AGENCIES, 2):
                corr_rows.append({"definition": definition, "grid_deg": width,
                                  "agency_left": left, "agency_right": right,
                                  "spatial_correlation": safe_corr(changes[(left, definition, width)],
                                                                   changes[(right, definition, width)])})
    pd.DataFrame(corr_rows).to_csv(out / "path_definition_agreement.csv", index=False)

    method = f"""# 路径密度定义敏感性

- 主口径：机构原生热带风暴及以上阶段的6小时轨迹点，经年度总和归一化。
- 风暴等权：每个风暴先在自身轨迹内归一化，再汇总并作年度归一化。
- 风暴—网格二元占用：每个风暴在每格至多计1次。
- 线段长度：相邻且间隔不超过12小时的轨迹段按大圆距离计权；每25 km以内细分后以子段中点归入网格，年度总长度归一化。
- 首次登陆前：有精确登陆交点的风暴截断至首次登陆时刻；未登陆风暴保留完整海上轨迹。
- 全生命史：对已达到机构原生热带风暴阈值的风暴，使用该机构全部原始位置报告，与仅热带风暴及以上阶段比较。
- 新口径均在2.5°网格作三机构检验；二元占用另作1°和5°分辨率检验；二元占用和登陆前路径另作2、3、4、5年块敏感性。既有轨迹点/风暴等权的三档网格和四档分块结果保留在`data/wnp_tc_robustness_matrix.csv`，不重复耗费计算。
- 每项均比较1966—1995与1996—2025，置换{cfg['n_permutations']}次，随机种子{cfg['random_seed']}；空间相关以三机构后期减前期差值图计算。
"""
    (out / "method.md").write_text(method, encoding="utf-8")
    print(results.groupby("definition")["block_permutation_p"].agg(["count", "min", "max"]).to_dict("index"))


if __name__ == "__main__":
    run()
