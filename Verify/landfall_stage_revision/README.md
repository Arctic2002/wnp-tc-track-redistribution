# 机构原生TS阶段登陆复核

## 目的

该目录隔离处理第三方审计指出的登陆阶段定义问题。它不改动正式Release、`CURRENT`、正文、图件或Handover。

现有精确交点来自“曾达到TS的风暴”的完整生命周期轨迹，因此某些交点本身可能发生在热带低压、变性或其他非TS阶段。复核在不重新求交的前提下，把当前正式版互斥岸段权威表中的精确交点与USA、JMA、CMA原生状态逐时次匹配。

## 预注册式口径

- 主口径：`pre_crossing_native_ts`，即精确交点前一个机构原生时次已经达到TS或以上。
- 宽松敏感性：`either_endpoint_native_ts`，即交点前后任一端达到TS或以上。
- 严格敏感性：`both_endpoints_native_ts`，即交点前后两端均达到TS或以上。
- 父结果复现：`full_lifecycle`，用于确认新脚本在不筛选阶段时逐项复现活动分析。
- 主要登陆定义：`first_landfall`；`all_events`作为事件加权的并列估计量。
- FDR：不同阶段规则、登陆定义、统计量和分析端点分别成族，每族只含USA、JMA、CMA三个检验。

主口径采用交点前状态，是因为登陆定义要求风暴在越过海岸线前已经处于TS阶段；宽松和严格口径用于显示6小时离散观测下的分类边界。

## 运行

在项目根目录执行：

```powershell
& '.venv\Scripts\python.exe' Verify\landfall_stage_revision\src\run_native_stage_landfall.py
```

## 输出

- `results/exact_events_with_native_stage.csv`：精确交点及原生阶段标记。
- `results/annual_metrics.csv`：四种阶段规则、两种登陆定义的年度统计。
- `results/period_statistics.csv`：时期差、块自助区间、3年分块置换p值及BH-FDR。
- `results/trend_statistics.csv`：1966年和1982年起的趋势敏感性。
- `results/leave_one_exclusive_coast.csv`：按互斥岸段执行的留一海岸检验。
- `results/title_gates.json`：原登记规则A与更严格首次登陆门禁并列结果。
- `CONCLUSION_IMPACT.md`：面向论文结论的中文说明。
- `figures/native_stage_effect_comparison.*`：全生命周期与主口径效应对照。
- `qa/`：输入哈希、父结果复现和结构门禁。

## 边界

本分析只修复直接登陆纬度的阶段定义。路径场、起源纬度Shapley分解、气候模态协变和旧聚类/回归均不在本目录重算。海岸份额是否同步采用TS阶段筛选作为单独作者决定，不在本轮自动扩展。
