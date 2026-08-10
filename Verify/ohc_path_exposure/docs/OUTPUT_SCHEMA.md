# 输出模式

| 文件 | 关键字段 | 用途 |
| --- | --- | --- |
| `ohc_region/oras5_ohc300_YYYY.nc` | `time, y, x, nav_lat, nav_lon, region_mask, ohc300` | 原生曲线网格区域逐月场 |
| `ohc_region_manifest.csv` | `year, stream, source_zip, source_sha256, months, grid_cells, output_sha256` | 区域裁切追溯 |
| `tracks_agency_definitions.csv.gz` | `agency, definition, sid, season, iso_time, lat, lon, wind` | 两种路径定义 |
| `track_extraction_qc.csv` | `agency, definition, year, storms, points` | 机构覆盖审计 |
| `matched_points.csv.gz` | 路径字段及 `ohc_year, ohc_month, point_is_land, grid_y, grid_x, grid_lat, grid_lon, match_km, ohc300, match_status` | 逐点配对、海陆标记与未匹配原因 |
| `match_qc.csv` | `agency, definition, track_season, calendar_year, total, matched, outside_domain, outside_ohc_time_window, too_far, missing_ohc_file, matched_land_points, matched_ocean_points` | 匹配率、跨年时次、陆上吸附与门限审计 |
| `decomposition_summary.csv` | 分期、机构、定义、年份口径、海陆口径、权重、`exposure_p1/p2, delta_exposure, redistribution_component, ocean_component, covariability_component, closure_residual` | 正文候选数值的唯一摘要源 |
| `decomposition_components.csv.gz` | `month, grid_y, grid_x, source_ocean_points, source_land_points, used_ocean_points_only, used_nearest_ocean_all_points, sensitivity_only_land_source_key, weights, fields, redistribution_cell, ocean_cell` | 分量空间追溯及状态键口径归属诊断 |
| `decomposition_bootstrap.csv` | 每个分量的 `q025, q50, q975` | 年份抽样不确定性 |
| `annual_exposure.csv` | `year, agency, definition, year_assignment, land_treatment, weighting, ohc300_exposure, storms, effective_storms, matched_points` | 年际暴露序列与时期均值复核；`storms`为出现有效片段的唯一气旋数，`effective_storms`为年度归一前的气旋权重总量 |
| `product_seam_diagnostics.csv`及`qa/product_seam_audit.json` | 2014/2015 结构、坐标容差/实际最大差值、有效足迹和数值诊断 | 产品拼接披露与人工裁定 |
| `output_manifest.csv` | `path, size_bytes, sha256` | 结果包完整性 |

所有结果表必须保留有量纲原值；图件只能从这些表派生，不能成为论文数值的唯一来源。
