"""公共工具：定位项目根目录、读取并校验配置、大圆距离、风速单位换算。"""
import yaml
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["periods"]
    rec = p.get("record_start", p["freq_start"])
    if not (p["env_start"] <= rec <= p["freq_start"] <= p["intensity_start"] <= p["end"]):
        raise ValueError(
            "periods 必须满足 env_start <= record_start <= freq_start <= intensity_start <= end")
    d = cfg["regions"]["dynamic"]
    if not (d["lon_min"] <= 90 and d["lon_max"] >= 180 and
            d["lat_min"] <= 10 and d["lat_max"] >= 60):
        raise ValueError("regions.dynamic 未完整覆盖副高指数定义域")
    for key in ("raw", "interim", "processed"):
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    cfg["gshhg_path"] = str((ROOT / cfg["gshhg_path"]).resolve())
    return cfg


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def kt_to_ms(kt):
    return kt * 0.514444


def ms_to_kt(ms):
    return ms / 0.514444
