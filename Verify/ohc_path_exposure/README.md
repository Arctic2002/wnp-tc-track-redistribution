# ORAS5 OHC300—热带气旋路径暴露验证包

本目录是独立的、尚未执行的代码审查包，用于检验西北太平洋热带气旋路径空间重分配是否改变了其所经海域的上层海洋热含量暴露。它不属于现行论文证据链，不能在第三方审查、调试和真实数据验收完成前回灌正文。

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

## 审查顺序

1. 审查 `config.json` 与 `docs/METHOD_SPEC.md` 中的估计对象和固定参数。
2. 审查机构原生记录筛选、海洋格点匹配和对称分解代码。
3. 审查合成测试是否覆盖已知答案、网格匹配、机构标记和产品拼接。
4. 将意见写入 `docs/THIRD_PARTY_REVIEW.md` 所述的独立目录，并完成裁定。
5. 审查通过后才建立 `qa/REVIEW_APPROVED.json`，再进入调试和真实数据执行。

`run_pipeline.py` 设有两级执行锁；当前目录只有示例文件，不能误启动正式流水线。代码复审批准后先运行 `prepare` 阶段；2014/2015 seam 产物完成人工裁定并以哈希绑定后，才允许运行 `analyze` 阶段。

`Verify/ocean_process` 的旧区域裁剪器不是本流水线输入。本包在隔离审查阶段只读取原始 ORAS5 ZIP；待本实现复审和调试通过后，再决定是否将旧入口改成薄封装，现阶段不互相覆盖。

## 预定入口（当前不要执行）

```powershell
python Verify\ohc_path_exposure\src\run_pipeline.py --phase prepare --execute
python Verify\ohc_path_exposure\src\run_pipeline.py --phase analyze --execute
```

依赖见 `requirements.txt`。所有派生结果仅写入本目录 `results/`，不会改写下载数据、Release、CURRENT 或 Handover。
