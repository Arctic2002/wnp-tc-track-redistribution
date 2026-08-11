# 路径型闭环独立验证

本目录包含“路径空间重分配如何传递到登陆纬度变化”的独立分析。分析使用固定路径类型、对称分解、条件标签置换、分块自助和域/分母敏感性检验。主要结果见`RESULTS_SUMMARY.md`。

## 分析范围

路径型分析包括：

1. 在主资料 1966—2025 年样本上固定三类路径模型；
2. 将路径密度变化精确拆为“类型比例变化”和“类型内部走廊位移”；
3. 对 `between_share`建立逐年条件标签置换零分布；
4. 对同一固定隶属度执行相对首个热带风暴点的路径分解；
5. 扫描 k=2—7 的 XB、FPC、种子稳定性和 `between_share`；
6. 比较 1966/1982 起算及预设移动切点；
7. 比较当前域内重归一、扩展至50°N和固定20点分母；
8. 在 USA、JMA、CMA 原生记录上使用同一模型复现；
9. 使用 3 年分块 bootstrap 给出固定模型下的年份抽样不确定性。

ERA5 月平均环流不提供逐气旋动力归因，因此不纳入该路径闭环分解。

## 目录

```text
pathway_closure/
├─ config.json                 固定分析口径
├─ docs/
│  ├─ METHOD_SPEC.md           方法、公式和判定规则
│  └─ INDEPENDENT_VALIDATION.md 独立验证标准
├─ src/
│  └─ run_pathway_closure.py   独立分析入口
├─ tests/
│  └─ test_exact_decomposition.py
├─ third_party/                独立验证材料
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

结果判读使用：

- `results/P0_ANALYSIS_REPORT.md`：前四项结果与边界；
- `results/between_share_permutation_summary.csv`：置换检验；
- `results/relative_genesis_summary.csv`：相对路径主分期；
- `results/k_scan_summary.csv`与`k_seed_diagnostics.csv`：类别数和种子稳定性；
- `results/start_cutpoint_sensitivity.csv`：起始年和切点规格结果；
- `results/domain_denominator_sensitivity.csv`：域边界、分母和对应置换结果；
- `qa/run_manifest.json`：输入哈希、环境版本和模型中心哈希。

## 结果使用边界

- `results/ANALYSIS_REPORT.md` 汇总验证结果。
- 代数闭合只作为实现不变量，不作为科学有效性判据。
- 跨机构一致只解释为对机构定位及原生口径差异不敏感，不解释为四次独立复现。
- 域边界与分母敏感性已完成；固定三类路径是统计分解框架，不表示唯一物理结构。
