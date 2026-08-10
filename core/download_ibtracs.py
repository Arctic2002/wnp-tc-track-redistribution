"""下载 IBTrACS 西北太平洋(WP)子集 CSV，原样落盘并写可追溯元数据。

产出：data/raw/ibtracs_wp.csv（+ .metadata.json）。
筛选全部留给 build_storm_table。
"""
# hashlib 计算文件指纹；json 保存下载元数据；datetime 记录下载时间。
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from core.utils import load_config

# 固定版本 URL 是可复现性的一部分；升级版本时应同时更新元数据说明。
URL = ("https://www.ncei.noaa.gov/data/"
       "international-best-track-archive-for-climate-stewardship-ibtracs/"
       "v04r01/access/csv/ibtracs.WP.list.v04r01.csv")


def main():
    """下载 IBTrACS 西北太平洋 CSV，并保存可追溯元数据。"""
    cfg = load_config()
    # skiprows=[1] 跳过 CSV 第二行的单位说明；它不是一条气旋观测。
    df = pd.read_csv(URL, skiprows=[1], low_memory=False)   # 第二行是单位行

    target = Path(cfg["paths"]["raw"]) / "ibtracs_wp.csv"
    df.to_csv(target, index=False)

    # SHA-256 相当于文件“指纹”：文件内容只要变化，哈希值就会变化。
    meta = {"url": URL, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "rows": len(df), "columns": list(df.columns)}
    # with_suffix 把 .csv 后缀替换为 .metadata.json。
    target.with_suffix(".metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("saved ibtracs_wp.csv", df.shape)


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main；被其他模块 import 时不会自动下载。
    main()
