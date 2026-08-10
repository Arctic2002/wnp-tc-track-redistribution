"""ENSO 位相环流合成差（El Niño − La Niña）的蒙特卡洛场显著性。

产出 processed/p2_fieldsig.npz（合成差场、逐点显著掩膜、场显著性 p 值）。
详见 Docs/02 §4.7。
观测与每次置换都只比较 El/La 且保持原组大小；零假设用固定块长的标签块重排近似
保留位相持续性；逐点阈值取零分布 2.5/97.5 分位；场统计量用余弦纬度加权显著面积；
p 值用加一校正。z500 读取 dynamic 框 ERA5。
"""
import os
# 先限制每个进程内部的数学库线程数，避免与外层并行争用核心。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import xarray as xr, pandas as pd, numpy as np
from joblib import Parallel, delayed
from core.utils import load_config


def composite_diff(field, labels):
    """根据标签计算 El Niño 年度场均值减 La Niña 年度场均值。"""
    years = field.year.values
    # 布尔索引从全部年份中取出两组年份编号；Neutral 不进入差值。
    ya, yb = years[labels == "El_Nino"], years[labels == "La_Nina"]
    return field.sel(year=ya).mean("year") - field.sel(year=yb).mean("year")


def block_shuffle(labels, block, rng):
    """对标签做「整块重排」：把相邻年份标签切成长度 block 的连续块，随机打乱块的
    顺序后拼回。块内顺序不变→保留位相的逐年持续性；打乱块的相对位置→打破标签
    与年份场的对应关系，构成零假设样本。只重排不改值，故 El/La/Neutral 各自总数
    严格不变；用整数索引打乱「块编号」再按编号取块（对非数值序列直接 shuffle
    在不同 numpy 版本下并不稳健）。
    """
    n = len(labels)
    nblocks = int(np.ceil(n / block))
    order = np.arange(nblocks)
    rng.shuffle(order)
    pieces = [labels[i * block:(i + 1) * block] for i in order]
    return np.concatenate(pieces)            # 拼回后长度=n、各标签计数不变


def main():
    cfg = load_config()
    st = xr.open_dataset(f"{cfg['paths']['interim']}/steering.nc")
    plev = xr.open_dataset(f"{cfg['paths']['interim']}/era5_wnp_dynamic_plev.nc")
    mon = cfg["typhoon_season"]
    # 字典让四个场共用同一套置换和输出逻辑。
    fields = {"u": st["u_steer"], "v": st["v_steer"], "shear": st["shear"],
              "z500": plev["z"].sel(level=500) / 9.80665}
    fields = {k: (v.sel(time=v.time.dt.month.isin(mon)).groupby("time.year").mean())
              for k, v in fields.items()}

    oni = pd.read_csv(f"{cfg['paths']['raw']}/indices/oni.csv")
    years = fields["u"].year.values
    phase = oni.set_index("season")["jas_oni"].reindex(years)
    # 只保留同时拥有环境场和 ONI 的共同年份。
    valid = phase.notna().to_numpy()
    years = years[valid]
    phase = phase.iloc[np.where(valid)[0]]
    fields = {k: v.sel(year=years) for k, v in fields.items()}
    th = cfg["oni_threshold"]
    labels = np.select([phase.to_numpy() >= th, phase.to_numpy() <= -th],
                       ["El_Nino", "La_Nina"], default="Neutral")
    nperm, nw = 1000, cfg["compute"]["n_workers"]
    block = cfg["statistics"]["bootstrap_block"]
    base_seed = cfg["statistics"]["random_seed"]
    # result 先存公共元数据，循环中再加入每个变量的观测差、掩膜和场 p 值。
    result = {"lon": fields["u"].longitude.values, "lat": fields["u"].latitude.values,
              "years": years, "labels": labels, "nperm": nperm, "block": block}

    for name, fy in fields.items():
        fy = fy.load()
        obs = composite_diff(fy, labels).values

        def run_batch(seeds):
            """在一个工作进程内连续完成一批随机置换，减少进程通信。"""
            out = []
            for seed in seeds:
                lab = block_shuffle(labels.copy(), block,
                                    np.random.default_rng(base_seed + int(seed)))
                out.append(composite_diff(fy, lab).values)
            return np.asarray(out)
        # array_split 把 1000 个种子尽量平均分给 nw 个进程。
        batches = np.array_split(np.arange(nperm), nw)
        null = np.concatenate(Parallel(n_jobs=nw)(delayed(run_batch)(b) for b in batches))
        # axis=0 表示对「置换次数」维求每个格点的 2.5% 和 97.5% 阈值。
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        obs_sig = (obs < lo) | (obs > hi)
        # [:,None] 把一维纬度权重变成「纬度×1」，以便自动广播到所有经度。
        area = np.cos(np.deg2rad(fy.latitude.values))[:, None]
        obs_area = float((obs_sig * area).sum())
        null_area = (((null < lo) | (null > hi)) * area).sum(axis=(1, 2))
        # 分子分母都加 1，避免有限置换次数下得到不合理的 p=0。
        field_p = float(((null_area >= obs_area).sum() + 1) / (nperm + 1))
        result.update({f"{name}_obs": obs, f"{name}_sig": obs_sig,
                       f"{name}_field_p": field_p})
    np.savez(f"{cfg['paths']['processed']}/p2_fieldsig.npz", **result)
    print("p2_fieldsig.npz written; field p-values: " +
          ", ".join(f"{k}={result[k+'_field_p']:.3f}" for k in fields))


if __name__ == "__main__":
    main()
