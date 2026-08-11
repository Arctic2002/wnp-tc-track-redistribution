# ORAS5 OHC300—热带气旋路径暴露分析

本目录包含西北太平洋热带气旋路径—季节重分配与同期月尺度 OHC300 暴露变化的分析代码、测试、派生结果和方法说明。完整分析覆盖 1966—2025 年，22 项合成测试全部通过。

## 固定边界

- 资料：ORAS5 0—300 m ocean heat content，1966—2025 年逐月场；不称为 TCHP。
- 区域：100°—180°E、0°—50°N；保留 ORAS5 原生曲线网格，不插值。
- 主分期：1966—1995 年与 1996—2025 年。
- 均一性敏感性：1982—2003 年与 2004—2025 年。
- 路径：PRIMARY、USA、JMA 和 CMA；后 3 套仅用于跨机构一致性/敏感性，不称为独立验证。
- 时间匹配：路径点与同年同月 OHC 配对，因此得到同期月尺度暴露，不是风暴前海洋环境，也不构成动力归因。
- 年份主口径：按 OHC 所属日历年归年；`season` 仅保留为风暴属性。敏感性剔除 `iso_time.year != season` 的跨年时次后按 `season` 归年。
- 海陆主口径：用 GSHHG 0.02°掩膜排除陆上源路径点；允许其吸附至 75 km 内最近海洋格点的旧口径只作敏感性。
- 估计对象：按“月份×ORAS5 原生海洋格点”构造暴露分布；重分配项同时含路径位置和发生月份构成变化，正文若使用必须写作“路径—季节暴露重分配”。

## 运行入口

```powershell
python Verify\ohc_path_exposure\src\run_pipeline.py --phase prepare --execute
python Verify\ohc_path_exposure\src\run_pipeline.py --phase analyze --execute
```

依赖见`requirements.txt`。派生结果存储在`results/`，整合结果存储在`integration/`，结果摘要见`RESULTS_SUMMARY.md`。
