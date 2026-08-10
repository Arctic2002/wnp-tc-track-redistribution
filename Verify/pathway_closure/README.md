# 路径型闭环独立验证

本目录用于检验“路径空间重分配如何传递到登陆纬度变化”。它与正式论文版本、`paper2_dynamic/` 主分析代码和 `data/` 数据目录隔离，不修改任何既有输入。

> 当前状态：前四项复核已执行，域边界/分母敏感性及科学裁决仍未闭环，暂不进入论文。见 `STATUS.md`。

## 当前范围

当前已完成的路径型复核：

1. 在主资料 1966—2025 年样本上固定三类路径模型；
2. 将路径密度变化精确拆为“类型比例变化”和“类型内部走廊位移”；
3. 对 `between_share`建立逐年条件标签置换零分布；
4. 对同一固定隶属度执行相对首个热带风暴点的路径分解；
5. 扫描 k=2—7 的 XB、FPC、种子稳定性和 `between_share`；
6. 比较 1966/1982 起算及预设移动切点；
7. 比较当前域内重归一、扩展至50°N和固定20点分母；
8. 在 USA、JMA、CMA 原生记录上使用同一模型复现；
9. 使用 3 年分块 bootstrap 给出固定模型下的年份抽样不确定性。

ERA5 月平均环流只适合后续检验动力一致性，不能提供逐气旋动力归因，因此本轮不把它混入路径闭环结果。

## 目录

```text
pathway_closure/
├─ config.json                 固定分析口径
├─ docs/
│  ├─ METHOD_SPEC.md           方法、公式和判定规则
│  └─ THIRD_PARTY_REVIEW.md    第三方代码与审查结果接入规则
├─ src/
│  └─ run_pathway_closure.py   独立分析入口
├─ tests/
│  └─ test_exact_decomposition.py
├─ third_party/                外部审查材料入口，不由本脚本改写
├─ results/                    分析结果与诊断图
└─ qa/                         运行清单和验证报告
```

## 运行

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe Verify\pathway_closure\src\run_pathway_closure.py
.\.venv\Scripts\python.exe Verify\pathway_closure\tests\test_exact_decomposition.py
```

脚本读取现有 `data/raw` 和 `data/processed`，所有写入均限制在本目录的 `results/` 与 `qa/`。

正式判读先读：

- `results/P0_ANALYSIS_REPORT.md`：前四项结果与边界；
- `results/between_share_permutation_summary.csv`：置换检验；
- `results/relative_genesis_summary.csv`：相对路径主分期；
- `results/k_scan_summary.csv`与`k_seed_diagnostics.csv`：类别数和种子稳定性；
- `results/start_cutpoint_sensitivity.csv`：起始年和切点规格结果；
- `results/domain_denominator_sensitivity.csv`：域边界、分母和对应置换结果；
- `qa/run_manifest.json`：输入哈希、环境版本和模型中心哈希。

## 结果使用边界

- `results/ANALYSIS_REPORT.md` 是验证性结果说明，不是论文正文。
- 代数闭合只作为实现不变量，不再作为科学入稿门禁。
- 跨机构一致只解释为对机构定位及原生口径差异不敏感，不解释为四次独立复现。
- 域边界与分母敏感性已完成；是否进入正式稿仍取决于路径类型框架的定位，不能把固定三类解释为唯一物理结构。
- 第三方结果须按 `docs/THIRD_PARTY_REVIEW.md` 登记来源和差异，不能直接覆盖本结果。
