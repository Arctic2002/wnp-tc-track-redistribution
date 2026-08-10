"""模糊 c 均值路径分型。

产出 processed/p2_clusters.csv（逐 TC 全部隶属度）与 p2_cluster_model.npz。
详见 Docs/02 §4.5。轨迹按真实相对时间连续插值到 20 点，经度 unwrap 并作纬度尺度
修正，特征标准化；k=3..7 和多初值下比较 Xie–Beni、FPC 与稳定性，保存全部隶属度。
"""
import pandas as pd, numpy as np, skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from core.utils import load_config

N = 20


def resample_track(g):
    """按真实相对时间把一条轨迹插值成 N 个经度点和 N 个纬度点。"""
    g = g.sort_values("iso_time")
    # 把时间转换成「距首点多少小时」，首值为 0。
    hours = (g.iso_time - g.iso_time.iloc[0]).dt.total_seconds().to_numpy() / 3600
    if hours[-1] <= 0:
        return None
    # q 是在真实生命期内均匀分布的 N 个目标时刻。
    q = np.linspace(0, hours[-1], N)
    lon = np.rad2deg(np.unwrap(np.deg2rad(g.lon.to_numpy())))
    return np.concatenate([np.interp(q, hours, lon), np.interp(q, hours, g.lat)])


def xb_index(X, cntr, u, m=2):
    """计算 Xie–Beni 指数；越小表示簇内紧凑且簇间分离越好。"""
    # cntr[:,None,:] 增加样本维，以广播方式计算「每个中心到每个样本」的距离。
    dist2 = ((cntr[:, None, :] - X.T[None, :, :]) ** 2).sum(axis=2)
    cd2 = ((cntr[:, None, :] - cntr[None, :, :]) ** 2).sum(axis=2)
    # 中心与自身距离为 0，不应参与「最小中心间距离」，所以替换为无穷大。
    cd2[cd2 == 0] = np.inf
    return float(((u ** m) * dist2).sum() / (X.shape[1] * cd2.min()))


def main():
    cfg = load_config()
    tr = pd.read_csv(f"{cfg['paths']['processed']}/tracks.csv", parse_dates=["iso_time"])
    tr = tr[tr.wind >= cfg["ts_threshold_kt"]].copy()
    if "nature" in tr:
        tr = tr[(tr.nature == "TS") | tr.nature.isna()]
    feats, sids = [], []
    for sid, g in tr.groupby("sid"):
        if len(g) >= 4:
            f = resample_track(g)
            if f is not None:
                feats.append(f)
                sids.append(sid)
    # raw 形状为「气旋数×40 特征」：前 20 列经度，后 20 列纬度。
    raw = np.vstack(feats)
    raw[:, :N] *= np.cos(np.deg2rad(20.0))               # WNP 参考纬度经度缩放
    # StandardScaler 记录每列均值和标准差，并把所有特征变成相近尺度。
    scaler = StandardScaler().fit(raw)
    X = scaler.transform(raw).T                           # features × samples

    best = None
    # k 尝试 3–7 类；每个 k 使用 20 个随机初值，降低局部最优风险。
    for k in range(3, 8):
        for seed in range(20):
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                X, k, m=2.0, error=1e-5, maxiter=1000, seed=seed)
            xb = xb_index(X, cntr, u)
            # best 元组第 0 项是 XB，因此只保留目前 XB 最小的候选结果。
            if best is None or xb < best[0]:
                best = (xb, fpc, k, cntr, u, seed)
    xb, fpc, k, cntr, u, seed = best
    # 按各簇加权平均生成经度排序，使重新运行后簇编号尽量保持稳定。
    order = np.argsort([np.average(raw[:, 0], weights=u[j]) for j in range(k)])
    cntr, u = cntr[order], u[order]                    # 标签按平均生成经度由西向东固定
    tab = pd.DataFrame({"sid": sids, "cluster": u.argmax(0), "membership": u.max(0)})
    # 除最大隶属度外，保留对每一簇的完整隶属度。
    for j in range(k):
        tab[f"membership_c{j}"] = u[j]
    tab.to_csv(f"{cfg['paths']['processed']}/p2_clusters.csv", index=False)
    np.savez(f"{cfg['paths']['processed']}/p2_cluster_model.npz", centers=cntr,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, k=k, xb=xb, fpc=fpc, seed=seed)
    print(f"p2_clusters.csv: {len(tab)} storms, k={k}, XB={xb:.3f}, FPC={fpc:.3f}; "
          f"sizes={np.bincount(tab.cluster, minlength=k).tolist()}")


if __name__ == "__main__":
    main()
