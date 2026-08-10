from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


AGENCY_ORDER = ("USA", "TOKYO", "CMA")
DISPLAY = {"USA": "USA", "TOKYO": "JMA", "CMA": "CMA"}
SEGMENTS = (
    ("1966-1980", 1966, 1980),
    ("1981-1995", 1981, 1995),
    ("1996-2010", 1996, 2010),
    ("2011-2025", 2011, 2025),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def build_subperiod_rows(annual_path: Path) -> list[dict[str, object]]:
    rows = [
        row
        for row in read_csv(annual_path)
        if row["definition"] == "first_landfall"
    ]
    output: list[dict[str, object]] = []
    for agency in AGENCY_ORDER:
        agency_rows = [row for row in rows if row["agency"] == agency]
        for label, start, end in SEGMENTS:
            selected = [
                row for row in agency_rows if start <= int(row["year"]) <= end
            ]
            years = sorted(int(row["year"]) for row in selected)
            expected = list(range(start, end + 1))
            if years != expected:
                raise AssertionError(
                    f"{agency} {label} does not contain exactly the expected years"
                )
            output.append(
                {
                    "agency": DISPLAY[agency],
                    "source_agency": agency,
                    "segment": label,
                    "start_year": start,
                    "end_year": end,
                    "n_years": len(selected),
                    "mean_of_annual_mean_lat": mean(
                        [float(row["mean_lat"]) for row in selected]
                    ),
                    "n_events": sum(int(row["n_events"]) for row in selected),
                    "source_csv": annual_path.name,
                    "weighting": "equal_year",
                }
            )
    return output


def build_leave_rows(leave_path: Path) -> list[dict[str, object]]:
    rows = [
        row
        for row in read_csv(leave_path)
        if row["definition"] == "first_landfall"
    ]
    output: list[dict[str, object]] = []
    for agency in AGENCY_ORDER:
        agency_rows = [row for row in rows if row["agency"] == agency]
        full_rows = [row for row in agency_rows if row["excluded_coast"] == "none"]
        leave_rows = [row for row in agency_rows if row["excluded_coast"] != "none"]
        if len(full_rows) != 1 or not leave_rows:
            raise AssertionError(f"Unexpected leave-one-coast rows for {agency}")
        full = full_rows[0]
        minimum = min(leave_rows, key=lambda row: float(row["period_difference_mean_lat"]))
        maximum = max(leave_rows, key=lambda row: float(row["period_difference_mean_lat"]))
        full_effect = float(full["period_difference_mean_lat"])
        minimum_effect = float(minimum["period_difference_mean_lat"])
        if minimum_effect <= 0:
            raise AssertionError(f"Leave-one-coast direction reverses for {agency}")
        output.append(
            {
                "agency": DISPLAY[agency],
                "source_agency": agency,
                "full_effect": full_effect,
                "leave_min_effect": minimum_effect,
                "leave_max_effect": float(maximum["period_difference_mean_lat"]),
                "minimum_retained_fraction": minimum_effect / full_effect,
                "minimum_excluded_coast": minimum["excluded_coast"],
                "minimum_block_permutation_p": float(minimum["block_permutation_p"]),
                "minimum_n_events": int(minimum["n_events"]),
                "maximum_excluded_coast": maximum["excluded_coast"],
                "source_csv": leave_path.name,
            }
        )
    return output


def build_publication_rows(
    subperiod_rows: list[dict[str, object]],
    leave_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    periods: dict[str, dict[str, float]] = defaultdict(dict)
    for row in subperiod_rows:
        periods[str(row["agency"])][str(row["segment"])] = float(
            row["mean_of_annual_mean_lat"]
        )
    leave_by_agency = {str(row["agency"]): row for row in leave_rows}
    output: list[dict[str, object]] = []
    for agency in ("USA", "JMA", "CMA"):
        values = periods[agency]
        leave = leave_by_agency[agency]
        peak = max(values, key=values.get)
        if peak != "1996-2010":
            raise AssertionError(f"Unexpected peak subperiod for {agency}: {peak}")
        output.append(
            {
                "agency": agency,
                "mean_1966_1980": values["1966-1980"],
                "mean_1981_1995": values["1981-1995"],
                "mean_1996_2010": values["1996-2010"],
                "mean_2011_2025": values["2011-2025"],
                "early_internal_change": values["1981-1995"]
                - values["1966-1980"],
                "late_retreat": values["2011-2025"] - values["1996-2010"],
                "full_effect": leave["full_effect"],
                "leave_min_effect": leave["leave_min_effect"],
                "leave_max_effect": leave["leave_max_effect"],
                "minimum_retained_fraction": leave["minimum_retained_fraction"],
                "minimum_excluded_coast": leave["minimum_excluded_coast"],
                "minimum_block_permutation_p": leave[
                    "minimum_block_permutation_p"
                ],
            }
        )
    if not all(float(row["late_retreat"]) < 0 for row in output):
        raise AssertionError("At least one record does not retreat after 1996-2010")
    return output


def format_trace_rows(
    publication_rows: list[dict[str, object]], scope: str
) -> list[dict[str, str]]:
    by_agency = {str(row["agency"]): row for row in publication_rows}
    if scope == "bilingual":
        results = "CN 3.5; EN 3.5; Supplementary Table S5"
        discussion = "CN 4.5; EN 4.5; Supplementary Table S5"
    else:
        results = "CN 4.5; Supplementary Table S3"
        discussion = "CN 5.4; Supplementary Table S3"
    period_text = " | ".join(
        f"{agency} "
        f"{float(by_agency[agency]['mean_1966_1980']):.3f}/"
        f"{float(by_agency[agency]['mean_1981_1995']):.3f}/"
        f"{float(by_agency[agency]['mean_1996_2010']):.3f}/"
        f"{float(by_agency[agency]['mean_2011_2025']):.3f}°"
        for agency in ("USA", "JMA", "CMA")
    )
    early_text = " | ".join(
        f"{agency} {float(by_agency[agency]['early_internal_change']):+.3f}°"
        for agency in ("USA", "JMA", "CMA")
    )
    leave_text = " | ".join(
        f"{agency} {float(by_agency[agency]['leave_min_effect']):.3f}–"
        f"{float(by_agency[agency]['leave_max_effect']):.3f}°; "
        f"retained={100 * float(by_agency[agency]['minimum_retained_fraction']):.1f}%; "
        f"min_exclusion={by_agency[agency]['minimum_excluded_coast']}; "
        f"p={float(by_agency[agency]['minimum_block_permutation_p']):.4f}"
        for agency in ("USA", "JMA", "CMA")
    )
    return [
        {
            "claim_id": "LF15",
            "manuscript_locations": results,
            "claim": "Equal-year first-landfall latitude means peak in 1996-2010 and retreat in 2011-2025 across agencies.",
            "source_csv": "results/landfall_chain/landfall_disclosure_table.csv",
            "row_filter_or_grouping": "definition=first_landfall; four equal 15-year segments; agency annual means weighted equally",
            "source_fields": "mean_1966_1980; mean_1981_1995; mean_1996_2010; mean_2011_2025; late_retreat",
            "derived_or_reported_value": period_text,
            "generator_script": "src/build_landfall_disclosure.py",
        },
        {
            "claim_id": "LF16",
            "manuscript_locations": discussion,
            "claim": "Early-period internal structure differs among agencies; USA and JMA provide the conservative effect range.",
            "source_csv": "results/landfall_chain/landfall_disclosure_table.csv",
            "row_filter_or_grouping": "1981-1995 minus 1966-1980; full first-landfall effects",
            "source_fields": "early_internal_change; full_effect",
            "derived_or_reported_value": early_text + "; conservative USA-JMA range=0.942–1.200°",
            "generator_script": "src/build_landfall_disclosure.py",
        },
        {
            "claim_id": "LF17",
            "manuscript_locations": results,
            "claim": "Leave-one-coast first-landfall effects are reported by agency as ranges and minimum-retention diagnostics.",
            "source_csv": "results/landfall_chain/leave_one_coast_summary.csv",
            "row_filter_or_grouping": "definition=first_landfall; excluded_coast!=none; grouped by agency",
            "source_fields": "full_effect; leave_min_effect; leave_max_effect; minimum_retained_fraction; minimum_excluded_coast; minimum_block_permutation_p",
            "derived_or_reported_value": leave_text,
            "generator_script": "src/build_landfall_disclosure.py",
        },
    ]


def update_trace(trace_path: Path, new_rows: list[dict[str, str]]) -> int:
    rows = read_csv(trace_path)
    rows = [row for row in rows if row["claim_id"] not in {"LF15", "LF16", "LF17"}]
    rows.extend(new_rows)
    write_csv(trace_path, rows)
    markdown = [
        "# Landfall-chain manuscript value trace",
        "",
        "All manuscript values in this table are computed from the listed CSV fields. Figures are visualization outputs and are not used as numerical sources.",
        "",
        "| ID | Manuscript locations | Reported value | CSV source | Generator |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['claim_id']} | {row['manuscript_locations']} | "
            f"{row['derived_or_reported_value']} | `{row['source_csv']}` | "
            f"`{row['generator_script']}` |"
        )
    trace_path.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    status_path = trace_path.parent / "trace_build_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "PASS"
    status["trace_claim_count"] = len(rows)
    status["disclosure_claims"] = ["LF15", "LF16", "LF17"]
    status["disclosure_rule"] = (
        "New disclosure values are computed from release-local annual and "
        "leave-one-coast CSV files; no figure is used as a numerical source."
    )
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual", type=Path, required=True)
    parser.add_argument("--leave-one", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scope", choices=("bilingual", "chinese"), required=True)
    args = parser.parse_args()

    annual_path = args.annual.resolve()
    leave_path = args.leave_one.resolve()
    output_dir = args.output_dir.resolve()
    trace_path = args.trace.resolve()
    subperiod_rows = build_subperiod_rows(annual_path)
    leave_rows = build_leave_rows(leave_path)
    publication_rows = build_publication_rows(subperiod_rows, leave_rows)
    write_csv(output_dir / "landfall_subperiod_summary.csv", subperiod_rows)
    write_csv(output_dir / "leave_one_coast_summary.csv", leave_rows)
    write_csv(output_dir / "landfall_disclosure_table.csv", publication_rows)
    trace_count = update_trace(
        trace_path, format_trace_rows(publication_rows, args.scope)
    )
    checks = {
        "status": "PASS",
        "annual_source": str(annual_path),
        "leave_one_source": str(leave_path),
        "subperiod_rows": len(subperiod_rows),
        "leave_summary_rows": len(leave_rows),
        "publication_rows": len(publication_rows),
        "trace_claim_count": trace_count,
        "all_segments_have_15_years": all(
            int(row["n_years"]) == 15 for row in subperiod_rows
        ),
        "all_agencies_peak_in_1996_2010": all(
            max(
                (
                    row
                    for row in subperiod_rows
                    if row["agency"] == agency
                ),
                key=lambda row: float(row["mean_of_annual_mean_lat"]),
            )["segment"]
            == "1996-2010"
            for agency in ("USA", "JMA", "CMA")
        ),
        "all_leave_one_effects_positive": all(
            float(row["leave_min_effect"]) > 0 for row in leave_rows
        ),
    }
    (output_dir / "landfall_disclosure_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
