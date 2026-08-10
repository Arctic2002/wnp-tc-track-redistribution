# paper2_dynamic

Paper II（动力线：路径与空间分布）分析层。脚本已建立，实际运行状态、依赖和已知限制见 `HANDOFF_paper2.md`；方法规范与图件规范分别见 `docs/02_paper2_dynamic.md` 和 `docs/03_figures_plan.md`。

## 2026-07-16 论文修订分析

当前题目为：**西北太平洋1966—2025年热带气旋路径的空间重分配与登陆格局变化**。

当前正文修订副本为 [`../Word/西北太平洋1966—2025年热带气旋路径的空间重分配与登陆格局变化_修订工作稿.docx`](../Word/西北太平洋1966—2025年热带气旋路径的空间重分配与登陆格局变化_修订工作稿.docx)，原始初稿保留不变。

三项补充分析已经完成：

- `revision_stats.py`：时间分块置换、BH-FDR、趋势与分解共用统计。
- `28_spatial_redistribution.py`：1966—1995 年与 1996—2025 年相对路径密度、风暴等权和登陆构成检验。
- `agency_data.py`、`29_multiagency_sensitivity.py`：USA/JMA/CMA 原始机构记录的路径、频数、LMI 纬度和登陆敏感性。
- `30_speed_decomposition_trial.py`：Feng（2024）对齐口径及论文原样本口径的移速分解试算。

结果写入 `revision_outputs/data/`，诊断图写入 `revision_outputs/figures/`。核心判断是：路径场整体重分配跨机构稳健；频数趋势对机构口径敏感；北部相对南部海岸份额上升，但八类海岸整体构成证据不一致；移速分解只作诊断。完整数值与写作边界见 `revision_outputs/三项补充分析结论与论文表述边界.md`。

主要模块按DAG顺序为：

- `20_track_metrics.py`：逐TC移速、转向、LMI纬度和登陆指标。
- `21_subtropical_high.py`：1940–2025年WNPSH面积、强度、脊线纬度和西伸脊点经度。
- `21b_monsoon_trough.py`：季风槽代理指标。
- `22_steering_flow.py`：引导气流与切变。
- `23_track_density.py`、`24_track_clustering.py`：路径密度和模糊c均值分型。
- `25_composites_regression.py`、`26_field_significance.py`：年度回归和合成场显著性。
- `27_figures.py`：P2-1至P2-14图件。
- `28_spatial_redistribution.py`：路径与登陆空间构成的场整体检验。
- `29_multiagency_sensitivity.py`：三机构敏感性检验。
- `30_speed_decomposition_trial.py`：移速分解诊断试算。
- `c1_density_triptych.py`至`c5_cluster_profiles.py`：补充气候学模块。

## Fig. 11 / P2-5时间口径

`21_subtropical_high.py`生成的1940–2025年ERA5序列全部保留。为与TC强度类资料的主分析期保持一致，`27_figures.py::fig05()`只用1982–2025年计算Theil–Sen趋势和移动块自助法95%置信区间；1940–1981年以灰色历史背景展示，不参与拟合。该截断用于跨变量比较，不表示ERA5从1982年才可用。

当前图件输出为：

- 投稿命名：`figures/Fig11_wnpsh.png/.pdf`。
- 管线命名：`figures/p2_fig05_wnpsh.png/.pdf`，与投稿命名副本内容一致。
- 趋势统计：`figures/Fig11_wnpsh_trend_stats.csv`。

1982–2025年每十年Theil–Sen斜率为：面积+2.20 × 10⁶ km²、强度指数+4.56、脊线纬度+0.30°、西伸脊点经度−7.03°，四项移动块自助法95%置信区间均不跨0。面积、强度和西伸脊点采用固定588 dagpm诊断，长期趋势可能包含背景位势高度整体升高的影响，不能单独解释为副高环流动力增强；图中区间也不等同于FDR趋势检验。

所有模块读取Core产物，并通过 `core.utils.load_config()` 取得配置。
