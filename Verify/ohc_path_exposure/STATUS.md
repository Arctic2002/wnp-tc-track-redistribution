# 状态

- 阶段：`EXECUTION_COMPLETE_QC_PASS_PENDING_RUNTIME_FIX_RECHECK`
- 新代码是否执行：是
- 合成测试是否执行：是，22项全部通过
- 真实 ORAS5/IBTrACS 分析是否执行：是，`prepare`与`analyze`均完成
- 结果整合是否完成：是，7张汇总表及`integration/INTEGRATED_RESULTS_REPORT_20260801.md`已生成并通过源清单、整合清单哈希核验
- 是否进入论文证据链：否
- 是否修改正式稿、Release、CURRENT 或 Handover：否

结果包共71个文件、381.87 MiB，输出清单哈希核验全部通过。运行时发现严格坐标逐位相等会把最大`3.0518×10⁻⁵°`的早期float32量化差误判为网格变化；修正为已披露的`5×10⁻⁵°`绝对容差后复跑成功，2014/2015坐标逐位一致且有效足迹零变化。完整过程见`qa/EXECUTION_REPORT_20260801.md`，科学整合见`integration/INTEGRATED_RESULTS_REPORT_20260801.md`。当前只待该运行时修复的第三方复核及是否回灌论文的作者决定。
