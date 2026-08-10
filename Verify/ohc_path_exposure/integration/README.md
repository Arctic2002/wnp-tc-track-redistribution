# OHC结果整合层

本目录只对`../results/`中的已完成CSV做可追溯汇总，不重新计算OHC场、路径匹配、分解或bootstrap，也不属于正式论文版本。

固定约束：

- 不计算或报告“重分配项占总变化比例”；
- `covariability_component`只作为代数共同变化项，不能解释为物理交互或反馈；
- 空间汇总是既有格点分量的精确加总，不把路径—季节重分配项拆称为纯路径项；
- 所有结论以CSV为依据，不从图读取数值。

执行入口：

```powershell
python Verify\ohc_path_exposure\integration\src\summarize_integrated_results.py --execute --overwrite
```

20°N南北分区图件入口：

```powershell
python Verify\ohc_path_exposure\integration\src\plot_ohc_latitude_contributions.py `
  --source Verify\ohc_path_exposure\integration\results\spatiotemporal_contributions.csv `
  --estimates Verify\ohc_path_exposure\integration\results\main_estimates.csv `
  --output-dir Verify\ohc_path_exposure\integration\figures
```

该入口在绘图前核对南北权重闭合，并核对分区加总与主配置`P`、`O`分量一致；同时输出中英文PNG/PDF及可审计的Flint语义规格。20°N只作为描述性加总界线，图中不提供未计算的分区级bootstrap区间。
