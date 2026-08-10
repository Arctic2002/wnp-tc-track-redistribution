"""Re-evaluate exact landfall crossings under agency-native TS definitions.

This isolated verification reads the current exclusive-coast authority and
original IBTrACS agency fields, but writes only below
``Verify/landfall_stage_revision``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
VERIFY = PROJECT / "Verify" / "landfall_stage_revision"
RESULTS = VERIFY / "results"
FIGURES = VERIFY / "figures"
QA = VERIFY / "qa"

JCLIMATE = PROJECT / "JClimate_Mainline"
RELEASE_TAG = (JCLIMATE / "CURRENT").read_text(encoding="utf-8").strip()
RELEASE = JCLIMATE / "releases" / RELEASE_TAG
RAW = PROJECT / "data" / "raw" / "IBTrACS.WP.v04r01.csv"
AUTHORITY = RELEASE / "results" / "exclusive_coast" / (
    "classified_exact_vector_events_admin0_corrected.csv"
)
ACTIVE_EXACT = JCLIMATE / "analysis" / "01_landfall_latitude" / (
    "landfall_events_exact.csv"
)
ACTIVE_SUMMARY = JCLIMATE / "analysis" / "01_landfall_latitude" / (
    "landfall_latitude_summary.csv"
)

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from JClimate_Mainline.src.stats import (  # noqa: E402
    bh_fdr,
    block_bootstrap_many,
    block_permutation_many,
    trend_summary,
)
from paper2_dynamic.agency_data import (  # noqa: E402
    AGENCIES,
    agency_flag,
    native_ts_mask,
    read_ibtracs_agencies,
)


START_YEAR = 1966
SPLIT_YEAR = 1996
END_YEARS = (2024, 2025)
BLOCK = 3
N_PERMUTATIONS = 9999
N_BOOTSTRAP = 4999
SEED = 202406

AGENCY_LABEL = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}
STAGE_RULES = (
    "full_lifecycle",
    "pre_crossing_native_ts",
    "either_endpoint_native_ts",
    "both_endpoints_native_ts",
)
PRIMARY_STAGE_RULE = "pre_crossing_native_ts"
DEFINITIONS = ("first_landfall", "all_events")
METRICS = ("mean_lat", "median_lat", "q25_lat", "q75_lat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def native_state_table(source: pd.DataFrame, agency: str) -> pd.DataFrame:
    """Return one agency-native state record for each usable position."""
    info = AGENCIES[agency]
    positioned = (
        agency_flag(source, agency, intensity=False)
        & source[info["lat"]].notna()
        & source[info["lon"]].notna()
    )
    state = source.loc[
        positioned,
        ["SID", "ISO_TIME", "NATURE", info["class"], info["wind"]],
    ].copy()
    state.columns = [
        "sid",
        "state_time",
        "nature",
        "native_class",
        "native_wind",
    ]
    state["native_ts"] = native_ts_mask(source, agency).loc[
        positioned
    ].to_numpy(dtype=bool)
    state["intensity_original"] = agency_flag(
        source, agency, intensity=True
    ).loc[positioned].to_numpy(dtype=bool)
    state = state.sort_values(["sid", "state_time"]).drop_duplicates(
        ["sid", "state_time"], keep="last"
    )
    if state.duplicated(["sid", "state_time"]).any():
        raise RuntimeError(f"Non-unique native state keys for {agency}")
    return state


def attach_native_stage(events: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Attach pre/post crossing native state and four explicit stage rules."""
    out = events.copy()
    for column in ("time", "segment_start", "segment_end"):
        out[column] = pd.to_datetime(out[column])
    if out["coast_exclusive"].isna().any():
        raise RuntimeError("Exclusive-coast authority has missing assignments")
    event_key = ["agency", "sid", "segment_start", "segment_end", "lat", "lon"]
    if out.duplicated(event_key).any():
        raise RuntimeError("Exact crossing authority has duplicate event keys")

    chunks = []
    fields = (
        "native_ts",
        "intensity_original",
        "native_class",
        "native_wind",
        "nature",
    )
    for agency in AGENCIES:
        group = out.loc[out["agency"].eq(agency)].copy()
        state = native_state_table(source, agency)
        for side, time_column in (
            ("start", "segment_start"),
            ("end", "segment_end"),
        ):
            renamed = state.rename(
                columns={
                    "state_time": time_column,
                    **{field: f"{side}_{field}" for field in fields},
                }
            )
            group = group.merge(
                renamed,
                on=["sid", time_column],
                how="left",
                validate="many_to_one",
            )
        chunks.append(group)
    out = pd.concat(chunks, ignore_index=True)

    required = [
        "start_native_ts",
        "end_native_ts",
        "start_intensity_original",
        "end_intensity_original",
    ]
    missing = {column: int(out[column].isna().sum()) for column in required}
    if any(missing.values()):
        raise RuntimeError(f"Unmatched endpoint states: {missing}")
    if not (
        out["start_intensity_original"].astype(bool)
        & out["end_intensity_original"].astype(bool)
    ).all():
        raise RuntimeError(
            "An exact segment endpoint lacks an original/verified agency "
            "intensity report"
        )

    out["full_lifecycle"] = True
    out["pre_crossing_native_ts"] = out["start_native_ts"].astype(bool)
    out["either_endpoint_native_ts"] = (
        out["start_native_ts"].astype(bool) | out["end_native_ts"].astype(bool)
    )
    out["both_endpoints_native_ts"] = (
        out["start_native_ts"].astype(bool) & out["end_native_ts"].astype(bool)
    )
    out["coast"] = out["coast_exclusive"]
    return out.sort_values(["agency", "season", "sid", "time"]).reset_index(
        drop=True
    )


def defined_events(events: pd.DataFrame, definition: str) -> pd.DataFrame:
    ordered = events.sort_values(["sid", "time"])
    if definition == "first_landfall":
        return ordered.drop_duplicates("sid", keep="first")
    if definition == "all_events":
        return ordered
    raise ValueError(definition)


def annual_metric_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = np.arange(START_YEAR, max(END_YEARS) + 1)
    for stage_rule in STAGE_RULES:
        staged = events.loc[events[stage_rule]]
        for agency in AGENCIES:
            agency_events = staged.loc[staged["agency"].eq(agency)]
            for definition in DEFINITIONS:
                sample = defined_events(agency_events, definition)
                for year in years:
                    annual = sample.loc[sample["season"].eq(year)]
                    values = annual["lat"].to_numpy(dtype=float)
                    rows.append(
                        {
                            "stage_rule": stage_rule,
                            "agency": agency,
                            "definition": definition,
                            "year": int(year),
                            "n_events": int(len(values)),
                            "n_storms": int(annual["sid"].nunique()),
                            "mean_lat": (
                                float(np.mean(values)) if len(values) else np.nan
                            ),
                            "median_lat": (
                                float(np.median(values)) if len(values) else np.nan
                            ),
                            "q25_lat": (
                                float(np.quantile(values, 0.25))
                                if len(values)
                                else np.nan
                            ),
                            "q75_lat": (
                                float(np.quantile(values, 0.75))
                                if len(values)
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def period_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combinations = [
        (stage_rule, agency, definition, metric)
        for stage_rule in STAGE_RULES
        for agency in AGENCIES
        for definition in DEFINITIONS
        for metric in METRICS
    ]
    for end_year in END_YEARS:
        years = np.arange(START_YEAR, end_year + 1)
        vectors = []
        for stage_rule, agency, definition, metric in combinations:
            series = (
                annual.loc[
                    annual["stage_rule"].eq(stage_rule)
                    & annual["agency"].eq(agency)
                    & annual["definition"].eq(definition)
                    & annual["year"].between(START_YEAR, end_year),
                    ["year", metric],
                ]
                .set_index("year")
                .reindex(years)[metric]
                .to_numpy(dtype=float)
            )
            vectors.append(series)
        matrix = np.column_stack(vectors)
        early = np.flatnonzero(years < SPLIT_YEAR)
        late = np.flatnonzero(years >= SPLIT_YEAR)
        difference, p_value = block_permutation_many(
            matrix,
            early,
            late,
            block=BLOCK,
            nperm=N_PERMUTATIONS,
            seed=SEED,
        )
        ci_low, ci_high = block_bootstrap_many(
            matrix[early],
            matrix[late],
            block=BLOCK,
            nboot=N_BOOTSTRAP,
            seed=SEED,
        )
        for i, (stage_rule, agency, definition, metric) in enumerate(
            combinations
        ):
            rows.append(
                {
                    "stage_rule": stage_rule,
                    "agency": agency,
                    "definition": definition,
                    "metric": metric,
                    "start": START_YEAR,
                    "end": end_year,
                    "period_difference": float(difference[i]),
                    "period_ci_low": float(ci_low[i]),
                    "period_ci_high": float(ci_high[i]),
                    "period_block_p": float(p_value[i]),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_family"] = (
        "landfall_stage_"
        + result["stage_rule"]
        + "_"
        + result["definition"]
        + "_"
        + result["metric"]
        + "_"
        + result["start"].astype(str)
        + "_"
        + result["end"].astype(str)
    )
    result["period_q_bh"] = np.nan
    for _, indices in result.groupby("fdr_family").groups.items():
        result.loc[indices, "period_q_bh"] = bh_fdr(
            result.loc[indices, "period_block_p"]
        )
    return result


def trend_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start_year, end_year in ((1966, 2025), (1982, 2025)):
        for stage_rule in STAGE_RULES:
            for agency in AGENCIES:
                for definition in DEFINITIONS:
                    for metric in METRICS:
                        subset = annual.loc[
                            annual["stage_rule"].eq(stage_rule)
                            & annual["agency"].eq(agency)
                            & annual["definition"].eq(definition)
                            & annual["year"].between(start_year, end_year)
                        ].sort_values("year")
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", RuntimeWarning)
                            trend = trend_summary(
                                subset["year"].to_numpy(),
                                subset[metric].to_numpy(),
                            )
                        rows.append(
                            {
                                "stage_rule": stage_rule,
                                "agency": agency,
                                "definition": definition,
                                "metric": metric,
                                "start": start_year,
                                "end": end_year,
                                **trend,
                                "trend_warning": " | ".join(
                                    str(item.message) for item in caught
                                ),
                            }
                        )
    result = pd.DataFrame(rows)
    result["fdr_family"] = (
        "landfall_stage_trend_"
        + result["stage_rule"]
        + "_"
        + result["definition"]
        + "_"
        + result["metric"]
        + "_"
        + result["start"].astype(str)
        + "_"
        + result["end"].astype(str)
    )
    result["mk_q_bh"] = np.nan
    for _, indices in result.groupby("fdr_family").groups.items():
        result.loc[indices, "mk_q_bh"] = bh_fdr(result.loc[indices, "mk_p"])
    return result


def leave_one_exclusive_coast(events: pd.DataFrame) -> pd.DataFrame:
    rows, vectors = [], []
    years = np.arange(START_YEAR, 2026)
    for stage_rule in ("full_lifecycle", PRIMARY_STAGE_RULE):
        staged = events.loc[events[stage_rule]]
        for agency in AGENCIES:
            agency_events = staged.loc[staged["agency"].eq(agency)]
            for definition in DEFINITIONS:
                sample = defined_events(agency_events, definition)
                for excluded_coast in [
                    "none",
                    *sorted(sample["coast"].unique()),
                ]:
                    selected = (
                        sample
                        if excluded_coast == "none"
                        else sample.loc[sample["coast"].ne(excluded_coast)]
                    )
                    vectors.append(
                        selected.groupby("season")["lat"]
                        .mean()
                        .reindex(years)
                        .to_numpy(dtype=float)
                    )
                    rows.append(
                        {
                            "stage_rule": stage_rule,
                            "agency": agency,
                            "definition": definition,
                            "excluded_coast": excluded_coast,
                            "n_events": int(len(selected)),
                            "n_storms": int(selected["sid"].nunique()),
                        }
                    )
    matrix = np.column_stack(vectors)
    early = np.flatnonzero(years < SPLIT_YEAR)
    late = np.flatnonzero(years >= SPLIT_YEAR)
    difference, p_value = block_permutation_many(
        matrix,
        early,
        late,
        block=BLOCK,
        nperm=N_PERMUTATIONS,
        seed=SEED,
    )
    result = pd.DataFrame(rows)
    result["period_difference_mean_lat"] = difference
    result["block_permutation_p"] = p_value
    return result


def title_gates(
    period: pd.DataFrame,
    trends: pd.DataFrame,
    leave_one: pd.DataFrame,
) -> dict:
    main = period.loc[
        period["stage_rule"].eq(PRIMARY_STAGE_RULE)
        & period["metric"].eq("mean_lat")
        & period["start"].eq(1966)
        & period["end"].eq(2025)
    ]
    no_2025 = period.loc[
        period["stage_rule"].eq(PRIMARY_STAGE_RULE)
        & period["metric"].eq("mean_lat")
        & period["start"].eq(1966)
        & period["end"].eq(2024)
    ]
    recent = trends.loc[
        trends["stage_rule"].eq(PRIMARY_STAGE_RULE)
        & trends["metric"].eq("mean_lat")
        & trends["start"].eq(1982)
        & trends["end"].eq(2025)
    ]
    leave = leave_one.loc[
        leave_one["stage_rule"].eq(PRIMARY_STAGE_RULE)
        & leave_one["excluded_coast"].ne("none")
    ]
    positive_share = (
        leave["period_difference_mean_lat"]
        .gt(0)
        .groupby([leave["agency"], leave["definition"]])
        .mean()
    )
    registered = {
        "stage_rule": PRIMARY_STAGE_RULE,
        "direction_all_first_and_all": bool(
            (main["period_difference"] > 0).all()
        ),
        "significant_agencies_in_either_definition": int(
            main.loc[main["period_q_bh"] < 0.05, "agency"].nunique()
        ),
        "no_2025_reversal_first_and_all": bool(
            (no_2025["period_difference"] > 0).all()
        ),
        "nonnegative_1982_2025_slope_first_and_all": bool(
            (recent["sen_slope_per_decade"] >= 0).all()
        ),
        "leave_one_coast_positive_share_ge_0_75_first_and_all": bool(
            positive_share.ge(0.75).all()
        ),
    }
    registered["passed"] = bool(
        registered["direction_all_first_and_all"]
        and registered["significant_agencies_in_either_definition"] >= 2
        and registered["no_2025_reversal_first_and_all"]
        and registered["nonnegative_1982_2025_slope_first_and_all"]
        and registered[
            "leave_one_coast_positive_share_ge_0_75_first_and_all"
        ]
    )

    first_main = main.loc[main["definition"].eq("first_landfall")]
    first_no_2025 = no_2025.loc[no_2025["definition"].eq("first_landfall")]
    first_recent = recent.loc[recent["definition"].eq("first_landfall")]
    first_leave = positive_share.loc[
        positive_share.index.get_level_values("definition") == "first_landfall"
    ]
    conventional = {
        "stage_rule": PRIMARY_STAGE_RULE,
        "definition": "first_landfall",
        "direction_all_agencies": bool(
            (first_main["period_difference"] > 0).all()
        ),
        "significant_agencies": int(
            first_main.loc[
                first_main["period_q_bh"] < 0.05, "agency"
            ].nunique()
        ),
        "no_2025_reversal": bool(
            (first_no_2025["period_difference"] > 0).all()
        ),
        "nonnegative_1982_2025_slope": bool(
            (first_recent["sen_slope_per_decade"] >= 0).all()
        ),
        "leave_one_coast_positive_share_ge_0_75": bool(
            first_leave.ge(0.75).all()
        ),
    }
    conventional["passed"] = bool(
        conventional["direction_all_agencies"]
        and conventional["significant_agencies"] >= 2
        and conventional["no_2025_reversal"]
        and conventional["nonnegative_1982_2025_slope"]
        and conventional["leave_one_coast_positive_share_ge_0_75"]
    )
    return {
        "registered_rule_A_recomputed": registered,
        "review_recommended_first_landfall_gate": conventional,
        "note": (
            "The second gate is a stricter review diagnostic and does not "
            "silently replace the registered project rule."
        ),
    }


def validate_against_active(
    events: pd.DataFrame, period: pd.DataFrame
) -> dict:
    active = pd.read_csv(
        ACTIVE_EXACT,
        parse_dates=["time", "segment_start", "segment_end"],
    )
    keys = ["agency", "sid", "segment_start", "segment_end"]
    merged = events[keys].drop_duplicates().merge(
        active[keys].drop_duplicates(),
        on=keys,
        how="outer",
        indicator=True,
    )
    event_key_sets_equal = bool(merged["_merge"].eq("both").all())

    active_summary = pd.read_csv(ACTIVE_SUMMARY)
    columns = [
        "agency",
        "definition",
        "period_difference",
        "period_ci_low",
        "period_ci_high",
        "period_block_p",
    ]
    active_key = active_summary.loc[
        active_summary["start"].eq(1966)
        & active_summary["end"].eq(2025)
        & active_summary["metric"].eq("mean_lat")
        & active_summary["definition"].isin(DEFINITIONS),
        columns,
    ]
    reproduced = period.loc[
        period["stage_rule"].eq("full_lifecycle")
        & period["start"].eq(1966)
        & period["end"].eq(2025)
        & period["metric"].eq("mean_lat")
        & period["definition"].isin(DEFINITIONS),
        columns,
    ]
    comparison = active_key.merge(
        reproduced,
        on=["agency", "definition"],
        suffixes=("_active", "_verify"),
        validate="one_to_one",
    )
    numeric_match = {}
    for column in (
        "period_difference",
        "period_ci_low",
        "period_ci_high",
        "period_block_p",
    ):
        numeric_match[column] = bool(
            np.allclose(
                comparison[f"{column}_active"],
                comparison[f"{column}_verify"],
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            )
        )
    comparison.to_csv(
        RESULTS / "active_full_lifecycle_reproduction.csv", index=False
    )
    passed = bool(
        event_key_sets_equal
        and not events["coast_exclusive"].isna().any()
        and all(numeric_match.values())
    )
    return {
        "authority_rows": int(len(events)),
        "active_exact_rows": int(len(active)),
        "event_key_sets_equal": event_key_sets_equal,
        "exclusive_coast_missing": int(events["coast_exclusive"].isna().sum()),
        "endpoint_state_missing": int(
            events[
                [
                    "start_native_ts",
                    "end_native_ts",
                    "start_intensity_original",
                    "end_intensity_original",
                ]
            ]
            .isna()
            .sum()
            .sum()
        ),
        "full_lifecycle_numeric_match": numeric_match,
        "passed": passed,
    }


def make_diagnostic_figure(period: pd.DataFrame) -> None:
    key = period.loc[
        period["stage_rule"].isin(["full_lifecycle", PRIMARY_STAGE_RULE])
        & period["definition"].isin(DEFINITIONS)
        & period["metric"].eq("mean_lat")
        & period["start"].eq(1966)
        & period["end"].eq(2025)
    ]
    colors = {
        "full_lifecycle": "#9A9A9A",
        PRIMARY_STAGE_RULE: "#2D6F9F",
    }
    labels = {
        "full_lifecycle": "Full lifecycle",
        PRIMARY_STAGE_RULE: "Native TS before crossing",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for ax, definition, title in zip(
        axes,
        DEFINITIONS,
        ("First qualifying landfall", "All qualifying landfall events"),
    ):
        subset = key.loc[key["definition"].eq(definition)]
        agencies = list(AGENCIES)
        x = np.arange(len(agencies))
        for offset, stage_rule in (
            (-0.12, "full_lifecycle"),
            (0.12, PRIMARY_STAGE_RULE),
        ):
            values = (
                subset.loc[subset["stage_rule"].eq(stage_rule)]
                .set_index("agency")
                .reindex(agencies)
            )
            y = values["period_difference"].to_numpy()
            ax.errorbar(
                x + offset,
                y,
                yerr=[
                    y - values["period_ci_low"].to_numpy(),
                    values["period_ci_high"].to_numpy() - y,
                ],
                fmt="o",
                capsize=3,
                color=colors[stage_rule],
                label=labels[stage_rule],
            )
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xticks(x, [AGENCY_LABEL[agency] for agency in agencies])
        ax.set_title(title)
        ax.set_ylabel("Late minus early annual mean latitude (°)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Effect of agency-native TS-stage restriction")
    fig.savefig(FIGURES / "native_stage_effect_comparison.png", dpi=300)
    fig.savefig(FIGURES / "native_stage_effect_comparison.pdf")
    plt.close(fig)


def write_impact_report(
    events: pd.DataFrame,
    period: pd.DataFrame,
    gates: dict,
    checks: dict,
) -> None:
    count_lines = [
        "| 机构 | 全生命周期交点 | 登陆前原生TS | 任一端为原生TS | 两端均为原生TS |",
        "|---|---:|---:|---:|---:|",
    ]
    for agency in AGENCIES:
        subset = events.loc[events["agency"].eq(agency)]
        count_lines.append(
            f"| {AGENCY_LABEL[agency]} | {len(subset)} | "
            f"{int(subset[PRIMARY_STAGE_RULE].sum())} | "
            f"{int(subset['either_endpoint_native_ts'].sum())} | "
            f"{int(subset['both_endpoints_native_ts'].sum())} |"
        )

    key = period.loc[
        period["stage_rule"].eq(PRIMARY_STAGE_RULE)
        & period["definition"].isin(DEFINITIONS)
        & period["metric"].eq("mean_lat")
        & period["start"].eq(1966)
        & period["end"].eq(2025)
    ].sort_values(["definition", "agency"])
    result_lines = [
        "| 定义 | 机构 | 后期−前期（°） | 95% CI | p | q（3机构同族） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in key.itertuples():
        result_lines.append(
            f"| {row.definition} | {AGENCY_LABEL[row.agency]} | "
            f"{row.period_difference:.3f} | "
            f"[{row.period_ci_low:.3f}, {row.period_ci_high:.3f}] | "
            f"{row.period_block_p:.4f} | {row.period_q_bh:.4f} |"
        )

    stage_sensitivity = period.loc[
        period["stage_rule"].isin(
            [
                PRIMARY_STAGE_RULE,
                "either_endpoint_native_ts",
                "both_endpoints_native_ts",
            ]
        )
        & period["metric"].eq("mean_lat")
        & period["start"].eq(1966)
        & period["end"].eq(2025)
    ]
    first_range = stage_sensitivity.loc[
        stage_sensitivity["definition"].eq("first_landfall"),
        "period_difference",
    ].agg(["min", "max"])
    all_range = stage_sensitivity.loc[
        stage_sensitivity["definition"].eq("all_events"),
        "period_difference",
    ].agg(["min", "max"])
    all_stage_directions_positive = bool(
        (stage_sensitivity["period_difference"] > 0).all()
    )

    registered = gates["registered_rule_A_recomputed"]
    conventional = gates["review_recommended_first_landfall_gate"]
    registered_text = "通过" if registered["passed"] else "未通过"
    conventional_text = "通过" if conventional["passed"] else "未通过"
    text = f"""# 机构原生TS阶段登陆复核：结论影响

## 复核对象

本分析不重新寻找交点，而是读取当前正式版 `{RELEASE_TAG}` 的6489个精确海岸线交点及其互斥岸段归属，再按USA、JMA和CMA各自原生强度状态判断交点是否发生在热带风暴及以上阶段。主口径为“交点前一个机构原生时次已达到TS”；“任一端为TS”和“两端均为TS”作为阶段判定敏感性。连续登陆纬度保留“其他”岸段，因为纬度指标不依赖命名海岸分类；留一海岸检验统一使用 `coast_exclusive`。

## 事件保留量

{chr(10).join(count_lines)}

## 主口径结果

{chr(10).join(result_lines)}

三种TS阶段判定下，三机构、两种登陆定义的时期差方向全部为正：`{all_stage_directions_positive}`。首次登陆的效应范围为{first_range['min']:.3f}°—{first_range['max']:.3f}°，全部事件为{all_range['min']:.3f}°—{all_range['max']:.3f}°。

## 标题门禁

- 原登记规则A重算：**{registered_text}**。方向、2025年端点、1982年起趋势、互斥岸段留一检验及“任一登陆定义达到显著”的机构数均按修正后的主口径重算。
- 更严格的首次登陆门禁：**{conventional_text}**。该诊断只使用首次登陆，并要求至少2个机构在三机构BH-FDR后显著；它是审稿视角下更保守的备选门禁，不在本轮擅自替换项目既定规则。
- 本轮不修改论文题名或正文。是否据更严格门禁调整题名，留待作者决定。

## 对现有结论的边界

- 路径场空间重分配和起源纬度分解不使用本次登陆阶段筛选，结论不受影响。
- 直接登陆纬度的方向、幅度和统计支持以本表为准；与全生命周期口径的差异见 `results/period_statistics.csv` 和诊断图。
- 海岸份额结果仍需另行决定是否也按TS阶段重算；本轮只修复直接登陆纬度及标题证据门禁，没有把阶段筛选自动外推到论文其他分析。
- 该结果是观测定义修复，不构成动力因果归因。

## 质量控制

- 互斥岸段权威表与活动精确交点键集合一致：`{checks['event_key_sets_equal']}`。
- 全生命周期回算与活动结果的效应量、区间和p值逐项一致：`{all(checks['full_lifecycle_numeric_match'].values())}`。
- 原生状态端点缺失数：`{checks['endpoint_state_missing']}`。
- 总门禁：`{'PASS' if checks['passed'] else 'FAIL'}`。
"""
    (VERIFY / "CONCLUSION_IMPACT.md").write_text(text, encoding="utf-8")


def main() -> None:
    for directory in (RESULTS, FIGURES, QA):
        directory.mkdir(parents=True, exist_ok=True)
    source = read_ibtracs_agencies(
        RAW, start=START_YEAR, end=max(END_YEARS)
    )
    authority = pd.read_csv(
        AUTHORITY,
        parse_dates=["time", "segment_start", "segment_end"],
        low_memory=False,
    )
    events = attach_native_stage(authority, source)
    events.to_csv(RESULTS / "exact_events_with_native_stage.csv", index=False)
    annual = annual_metric_rows(events)
    annual.to_csv(RESULTS / "annual_metrics.csv", index=False)
    period = period_statistics(annual)
    period.to_csv(RESULTS / "period_statistics.csv", index=False)
    trends = trend_statistics(annual)
    trends.to_csv(RESULTS / "trend_statistics.csv", index=False)
    leave_one = leave_one_exclusive_coast(events)
    leave_one.to_csv(
        RESULTS / "leave_one_exclusive_coast.csv", index=False
    )
    gates = title_gates(period, trends, leave_one)
    write_json(RESULTS / "title_gates.json", gates)

    checks = validate_against_active(events, period)
    write_json(QA / "validation_checks.json", checks)
    pd.DataFrame(
        [
            {
                "role": role,
                "path": str(path.relative_to(PROJECT)),
                "sha256": sha256(path),
            }
            for role, path in (
                ("raw_ibtracs", RAW),
                ("exclusive_coast_authority", AUTHORITY),
                ("active_exact_crossings", ACTIVE_EXACT),
                ("active_landfall_summary", ACTIVE_SUMMARY),
            )
        ]
    ).to_csv(QA / "input_hashes.csv", index=False)
    write_json(
        QA / "run_summary.json",
        {
            "release_authority": RELEASE_TAG,
            "primary_stage_rule": PRIMARY_STAGE_RULE,
            "stage_rules": list(STAGE_RULES),
            "definitions": list(DEFINITIONS),
            "metrics": list(METRICS),
            "period": {
                "early": "1966-1995",
                "late": "1996-2025",
                "endpoint_sensitivity": 2024,
            },
            "block_years": BLOCK,
            "n_permutations": N_PERMUTATIONS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "trend_warning_rows": int(
                trends["trend_warning"].fillna("").ne("").sum()
            ),
            "qa_passed": checks["passed"],
        },
    )
    make_diagnostic_figure(period)
    write_impact_report(events, period, gates, checks)
    if not checks["passed"]:
        raise RuntimeError(
            "Verification gates failed; inspect qa/validation_checks.json"
        )


if __name__ == "__main__":
    main()
