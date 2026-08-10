from __future__ import annotations

import numpy as np
import pandas as pd

from .common import REPORTS, WORK, ensure_output_dirs, load_config
from .landfall_latitude import AGENCY_LABEL, ANALYSIS, annual_metrics, definitions, summarize


def make_annual(events: pd.DataFrame) -> pd.DataFrame:
    years = np.arange(1966, 2026)
    rows = []
    for agency, group in events.groupby("agency"):
        for definition, sample in definitions(group).items():
            for row in annual_metrics(sample, years):
                rows.append({"agency": agency, "definition": definition, **row})
    return pd.DataFrame(rows)


def run():
    ensure_output_dirs()
    cfg = load_config()
    approximate = pd.read_csv(WORK / "data/upstream_revision/p2_multiagency_landfalls.csv", parse_dates=["time"])
    approximate = approximate.loc[approximate.season.between(1966, 2025)].copy()
    exact = pd.read_csv(ANALYSIS / "landfall_events_exact.csv", parse_dates=["time"])
    unresolved = pd.read_csv(ANALYSIS / "unresolved_candidates.csv", parse_dates=["time"])
    unresolved_key = unresolved[["agency", "sid", "time"]].drop_duplicates()
    unresolved_events = approximate.merge(unresolved_key, on=["agency", "sid", "time"], how="inner")
    hybrid = pd.concat([exact, unresolved_events], ignore_index=True, sort=False).sort_values(["agency", "sid", "time"])
    hybrid = hybrid.drop_duplicates(["agency", "sid", "time"])

    summaries = []
    for source, events in [("all_raster_candidates", approximate), ("exact_plus_unresolved_raster", hybrid)]:
        summary = summarize(make_annual(events), cfg)
        summary.insert(0, "source", source)
        summaries.append(summary)
    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(ANALYSIS / "candidate_screen_sensitivity_summary.csv", index=False)

    approximate["period"] = np.where(approximate.season <= 1995, "early", "late")
    unresolved_events["period"] = np.where(unresolved_events.season <= 1995, "early", "late")
    counts = approximate.groupby(["agency", "period"]).agg(candidate_n=("sid", "size"), candidate_mean_lat=("lat", "mean")).reset_index()
    missing = unresolved_events.groupby(["agency", "period"]).agg(unresolved_n=("sid", "size"), unresolved_mean_lat=("lat", "mean")).reset_index()
    audit = counts.merge(missing, on=["agency", "period"], how="left").fillna({"unresolved_n": 0})
    audit["unresolved_fraction"] = audit.unresolved_n / audit.candidate_n
    audit.to_csv(ANALYSIS / "unresolved_candidate_audit.csv", index=False)

    key = out.loc[(out.start == 1966) & (out.end == 2025) & out.metric.eq("mean_lat") & out.definition.isin(["first_landfall", "all_events"])]
    exact_key = pd.read_csv(ANALYSIS / "landfall_latitude_summary.csv")
    exact_key = exact_key.loc[(exact_key.start == 1966) & (exact_key.end == 2025) & exact_key.metric.eq("mean_lat") & exact_key.definition.isin(["first_landfall", "all_events"])]
    all_positive = bool((key.period_difference > 0).all() and (exact_key.period_difference > 0).all())
    lines = ["\n## 候选筛选与未解析事件敏感性\n", f"- 全部敏感性口径的时期差均为正：`{all_positive}`。"]
    for r in key.itertuples():
        lines.append(f"- {r.source} / {AGENCY_LABEL[r.agency]} / {r.definition}: Δ={r.period_difference:.3f}°，p={r.period_block_p:.4f}，q={r.period_q_bh:.4f}。")
    lines.append("- 精确交点表不以近似点填补；上述两套结果仅用于判断2.17%未解析候选是否足以改变方向。")
    with (REPORTS / "title_evidence_boundary.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(key[["source", "agency", "definition", "period_difference", "period_block_p", "period_q_bh"]].to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    run()
