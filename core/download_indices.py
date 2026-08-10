"""把 ONI、PDO、GSHHG 也纳入可复现下载链。

产出：data/raw/indices/oni.csv（season,jas_oni）、pdo.csv（season,pdo），
      以及把 GSHHG 发行包解压到 gshhg_path 指向的位置。
URL/格式以官方页面当时为准；解析失败显式报错并打印原文前几行，绝不静默写空表。
"""
# requests 下载文本/二进制；io/zipfile 处理压缩包；pandas 解析与落盘。
import io
import json
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from core.utils import load_config

# 这些 URL 会随官方改版而变化；运行前请到官方页面确认最新地址与列格式。
ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
PDO_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"


def _save_meta(path, url, extra=None):
    """与 download_ibtracs 一致：为每个落盘文件写来源/时间/哈希元数据，保证可追溯。"""
    meta = {"url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if extra:
        meta.update(extra)
    Path(path).with_suffix(".metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_oni(cfg):
    """下载并规范化 ONI（用 Nino3.4 月值聚合成 JAS 三月平均作为 jas_oni）。"""
    txt = requests.get(ONI_URL, timeout=60).text
    # 用空白分隔解析；首行是表头。解析失败会暴露原文前几行。
    try:
        df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
        df.columns = [c.strip().upper() for c in df.columns]
        # 兼容不同表头命名：年份列 YR、月份列 MON、距平列 ANOM。
        ycol = next(c for c in df.columns if c in ("YR", "YEAR"))
        mcol = next(c for c in df.columns if c in ("MON", "MONTH", "MM"))
        acol = next(c for c in df.columns if "ANOM" in c)
    except Exception as e:
        raise RuntimeError("ONI 解析失败，请核对格式。原文前 5 行：\n" +
                           "\n".join(txt.splitlines()[:5])) from e
    jas = df[df[mcol].isin([7, 8, 9])]
    out = (jas.groupby(ycol)[acol].mean().rename("jas_oni")
              .rename_axis("season").reset_index())
    target = Path(cfg["paths"]["raw"]) / "indices" / "oni.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    _save_meta(target, ONI_URL, {"rows": len(out)})
    return out


def fetch_pdo(cfg):
    """下载并规范化 PDO 月指数，聚合为台风季(6-10 月)年均 pdo。"""
    txt = requests.get(PDO_URL, timeout=60).text
    # PDO .dat 通常是 Year + 12 列月值的宽表，文件头有若干说明行。
    rows = []
    for line in txt.splitlines():
        parts = line.split()
        # 只接受“首列是 4 位年份且其后有 12 个可解析浮点”的数据行。
        if len(parts) >= 13 and parts[0].isdigit() and len(parts[0]) == 4:
            try:
                rows.append([int(parts[0])] + [float(x) for x in parts[1:13]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("PDO 解析失败，请核对格式。原文前 5 行：\n" +
                           "\n".join(txt.splitlines()[:5]))
    wide = pd.DataFrame(rows, columns=["season"] + list(range(1, 13)))
    season_months = cfg["typhoon_season"]
    out = wide.set_index("season")[season_months].mean(axis=1).rename("pdo").reset_index()
    target = Path(cfg["paths"]["raw"]) / "indices" / "pdo.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    _save_meta(target, PDO_URL, {"rows": len(out)})
    return out


def fetch_gshhg(cfg, zip_url):
    """下载并解压 GSHHG，校验目标 shp 是否就位。zip_url 由官方发行页给出。"""
    dst = Path(cfg["gshhg_path"]).parent          # gshhg_path 指向 GSHHS_h_L1.shp
    dst.mkdir(parents=True, exist_ok=True)
    blob = requests.get(zip_url, timeout=300).content
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dst.parent)                  # 解到 data/raw/ 下，保留发行目录结构
    if not Path(cfg["gshhg_path"]).exists():
        raise FileNotFoundError(
            f"解压后未找到 {cfg['gshhg_path']}；请核对 zip 内目录结构或调整 gshhg_path")


def _check_coverage(df, name, cfg):
    """覆盖度校验：指数年份必须覆盖频率/路径时段，否则下游 merge 会静默丢年份。"""
    need_lo, need_hi = cfg["periods"]["freq_start"], cfg["periods"]["end"]
    have = set(df["season"].astype(int))
    missing = [y for y in range(need_lo, need_hi + 1) if y not in have]
    if missing:
        print(f"[警告] {name} 缺少年份: {missing[:10]}{'...' if len(missing) > 10 else ''}")


def main(gshhg_zip_url=None):
    cfg = load_config()
    oni = fetch_oni(cfg)
    _check_coverage(oni, "ONI", cfg)
    pdo = fetch_pdo(cfg)
    _check_coverage(pdo, "PDO", cfg)
    if gshhg_zip_url:                              # GSHHG 体积大，按需传入官方 zip 地址
        fetch_gshhg(cfg, gshhg_zip_url)
    print("indices saved:", {"oni": len(oni), "pdo": len(pdo)})


if __name__ == "__main__":
    main()
