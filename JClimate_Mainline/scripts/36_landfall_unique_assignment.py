from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, PROJECT, block_permutation_scalar, ensure_dirs, period_indices
from paper2_dynamic.revision_stats import compositional_change_test

YEARS = np.arange(1966, 2026)
COASTS = ["China_E", "China_S", "Taiwan", "Japan", "Korea", "Philippines", "Vietnam", "Other"]
NORTH = {"China_E", "Taiwan", "Korea", "Japan"}
SOUTH = {"China_S", "Vietnam", "Philippines"}


def assign(group, rule):
    g = group.copy()
    g["time"] = pd.to_datetime(g["time"], errors="coerce")
    if rule == "first_any":
        return g.sort_values("time").iloc[0]
    if rule == "first_named":
        named = g.loc[g["coast"] != "Other"].sort_values("time")
        return named.iloc[0] if len(named) else g.sort_values("time").iloc[0]
    if rule == "strongest":
        valid = g.loc[g["wind"].notna()]
        if valid.empty:
            return None
        return valid.loc[valid["wind"].idxmax()]
    raise ValueError(rule)


def unique_table(events, by, rule):
    rows = []
    for keys, group in events.groupby(by, sort=False):
        row = assign(group, rule)
        if row is None:
            continue
        item = row.to_dict()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(by, keys):
            item[col] = value
        item["assignment_rule"] = rule
        rows.append(item)
    return pd.DataFrame(rows)


def annual_composition(d):
    rows = []
    north_share = []
    for year in YEARS:
        g = d.loc[d["season"] == year]
        c = g["coast"].value_counts().reindex(COASTS, fill_value=0)
        total = int(c.sum())
        if total == 0:
            raise ValueError(f"no assigned landfall events in {year}")
        rows.append((c / total).to_numpy(float))
        n = int(c.reindex(list(NORTH), fill_value=0).sum())
        s = int(c.reindex(list(SOUTH), fill_value=0).sum())
        if n + s == 0:
            raise ValueError(f"no named north/south landfall events in {year}")
        north_share.append(n / (n + s))
    return np.asarray(rows), np.asarray(north_share)


def summarize(d, agency, rule):
    comp, share = annual_composition(d)
    e, l = period_indices(YEARS)
    full = compositional_change_test(comp, e, l, nperm=9999, block=3, seed=202406)
    diff, p = block_permutation_scalar(share, e, l, block=3, nperm=9999)
    return {
        "agency": agency,
        "assignment_rule": rule,
        "n_unique_storms": int(d["sid"].nunique()),
        "n_rows": len(d),
        "eight_category_tv": full["tv"],
        "eight_category_block_p": full["global_p"],
        "early_north_named_share": float(share[e].mean()),
        "late_north_named_share": float(share[l].mean()),
        "north_share_change_percentage_points": diff * 100,
        "north_share_block_p": p,
    }


def run() -> None:
    ensure_dirs()
    storms = pd.read_csv(PROJECT / "data" / "processed" / "storms.csv", usecols=["sid", "season"])
    primary = pd.read_csv(PROJECT / "data" / "processed" / "landfalls.csv").merge(
        storms, on="sid", how="left", validate="many_to_one"
    )
    primary = primary.loc[primary["season"].between(1966, 2025)].copy()
    multi = pd.read_csv(DATA / "upstream_revision" / "p2_multiagency_landfalls.csv")
    multi = multi.loc[multi["season"].between(1966, 2025)].copy()

    outputs, summaries = [], []
    for rule in ("first_any", "first_named", "strongest"):
        d = unique_table(primary, ["sid"], rule)
        d["agency"] = "PRIMARY"
        outputs.append(d)
        summaries.append(summarize(d, "PRIMARY", rule))
    for agency, events in multi.groupby("agency"):
        for rule in ("first_any", "first_named"):
            d = unique_table(events, ["sid"], rule)
            d["agency"] = agency
            outputs.append(d)
            summaries.append(summarize(d, agency, rule))

    out = pd.concat(outputs, ignore_index=True)
    out.to_csv(DATA / "jcli_landfall_unique_events.csv", index=False)
    summary = pd.DataFrame(summaries)
    missing_wind_storms = primary.groupby("sid")["wind"].apply(lambda x: x.notna().sum() == 0).sum()
    summary["primary_storms_without_any_landfall_wind"] = int(missing_wind_storms)
    summary.to_csv(DATA / "jcli_landfall_unique_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
