# 机构原生TS阶段登陆分析

## 目的

本分析检验登陆交点是否发生在机构原生记录的热带风暴及以上阶段。

精确交点来自“曾达到TS的风暴”的完整生命周期轨迹，因此某些交点本身可能发生在热带低压、变性或其他非TS阶段。本分析不重新求交，而是将互斥岸段表中的精确交点与USA、JMA、CMA原生状态逐时次匹配。

## 预注册式口径

- 主口径：`pre_crossing_native_ts`，即精确交点前一个机构原生时次已经达到TS或以上。
- 宽松敏感性：`either_endpoint_native_ts`，即交点前后任一端达到TS或以上。
- 严格敏感性：`both_endpoints_native_ts`，即交点前后两端均达到TS或以上。
- 全生命周期参照：`full_lifecycle`，用于核对阶段筛选前后的结果差异。
- 主要登陆定义：`first_landfall`；`all_events`作为事件加权的并列估计量。
- FDR：不同阶段规则、登陆定义、统计量和分析端点分别成族，每族只含USA、JMA、CMA三个检验。

主口径采用交点前状态，是因为登陆定义要求风暴在越过海岸线前已经处于TS阶段；宽松和严格口径用于显示6小时离散观测下的分类边界。

## 运行

在项目根目录执行：

```powershell
& '.venv\Scripts\python.exe' Verify\native_stage_landfall\src\run_native_stage_landfall.py
```

## 输出

- `results/exact_events_with_native_stage.csv`：精确交点及原生阶段标记。
- `results/annual_metrics.csv`：四种阶段规则、两种登陆定义的年度统计。
- `results/period_statistics.csv`：时期差、块自助区间、3年分块置换p值及BH-FDR。
- `results/trend_statistics.csv`：1966年和1982年起的趋势敏感性。
- `results/leave_one_exclusive_coast.csv`：按互斥岸段执行的留一海岸检验。
- `results/title_criteria.json`：规则A与更严格首次登陆判据的并列结果。
- `RESULTS_SUMMARY.md`：结果与解释边界摘要。
- `figures/native_stage_effect_comparison.*`：全生命周期与主口径效应对照。
- `qa/`：输入哈希、参照结果核对和结构检查。

## 边界

本分析限于直接登陆纬度的阶段定义。路径场、起源纬度 Shapley 分解、气候模态协变、聚类和回归不使用该阶段筛选。海岸份额结果不在本分析的估计范围内。
