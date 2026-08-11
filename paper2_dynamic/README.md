# paper2_dynamic

本目录包含西北太平洋热带气旋路径、空间分布及其环流背景的分析模块。

## 分析模块

- `20_track_metrics.py`：逐TC移速、转向、LMI纬度和登陆指标。
- `21_subtropical_high.py`：1940–2025年WNPSH面积、强度、脊线纬度和西伸脊点经度。
- `21b_monsoon_trough.py`：季风槽代理指标。
- `22_steering_flow.py`：引导气流与切变。
- `23_track_density.py`、`24_track_clustering.py`：路径密度和模糊c均值分型。
- `25_composites_regression.py`、`26_field_significance.py`：年度回归和合成场显著性。
- `27_figures.py`：动力线图件。
- `28_spatial_redistribution.py`：路径与登陆空间构成的场整体检验。
- `29_multiagency_sensitivity.py`：三机构敏感性检验。
- `30_speed_decomposition_trial.py`：移速分解诊断。
- `c1_density_triptych.py`至`c5_cluster_profiles.py`：补充气候学分析。

## 时间口径

`21_subtropical_high.py`生成1940—2025年ERA5序列。`27_figures.py::fig05()`使用1982—2025年计算Theil—Sen趋势和移动块自助法95%置信区间，1940—1981年作为历史背景展示，不参与拟合。该时间划分用于与热带气旋强度资料的共同覆盖期比较，不表示ERA5从1982年才可用。

1982–2025年每十年Theil–Sen斜率为：面积+2.20 × 10⁶ km²、强度指数+4.56、脊线纬度+0.30°、西伸脊点经度−7.03°，四项移动块自助法95%置信区间均不跨0。面积、强度和西伸脊点采用固定588 dagpm诊断，长期趋势可能包含背景位势高度整体升高的影响，不能单独解释为副高环流动力增强；图中区间也不等同于FDR趋势检验。

所有模块读取Core产物，并通过`core.utils.load_config()`取得配置。
