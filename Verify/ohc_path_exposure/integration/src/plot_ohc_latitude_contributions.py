from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


COMPARISONS = [
    "1966_1995_vs_1996_2025",
    "1982_2003_vs_2004_2025",
]
ZONES = ["south_of_20N", "20N_and_north"]
COLORS = {
    "south_of_20N": "#E07B39",
    "20N_and_north": "#4C78A8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(source: Path, estimates_path: Path) -> pd.DataFrame:
    data = pd.read_csv(source)
    selected = data[
        (data["agency"] == "PRIMARY")
        & (data["aggregation"] == "latitude_zone")
        & (data["comparison"].isin(COMPARISONS))
        & (data["bin"].isin(ZONES))
    ].copy()
    if len(selected) != 4:
        raise ValueError(f"Expected four PRIMARY latitude-zone rows, found {len(selected)}")
    if selected.groupby("comparison")["bin"].nunique().to_dict() != {
        comparison: 2 for comparison in COMPARISONS
    }:
        raise ValueError("Each comparison must contain exactly two latitude zones")

    estimates = pd.read_csv(estimates_path)
    principal = estimates[
        (estimates["agency"] == "PRIMARY")
        & (estimates["definition"] == "ts_only")
        & (estimates["year_assignment"] == "calendar_year")
        & (estimates["land_treatment"] == "ocean_points_only")
        & (estimates["weighting"] == "storm_normalized_equal_year")
    ]
    for comparison in COMPARISONS:
        rows = selected[selected["comparison"] == comparison]
        if abs(rows["weight_change"].sum()) > 1e-12:
            raise ValueError(f"Weight closure failed for {comparison}")
        expected_p = principal[
            (principal["comparison"] == comparison)
            & (principal["component"] == "redistribution_component")
        ]["estimate_j_m2"].iloc[0]
        expected_o = principal[
            (principal["comparison"] == comparison)
            & (principal["component"] == "ocean_component")
        ]["estimate_j_m2"].iloc[0]
        if not np.isclose(rows["redistribution_cell"].sum(), expected_p, atol=1e-5, rtol=1e-12):
            raise ValueError(f"P closure failed for {comparison}")
        if not np.isclose(rows["ocean_cell"].sum(), expected_o, atol=1e-5, rtol=1e-12):
            raise ValueError(f"O closure failed for {comparison}")

    selected["weight_change_pp"] = selected["weight_change"] * 100.0
    selected["p_1e8"] = selected["redistribution_cell"] / 1e8
    selected["o_1e8"] = selected["ocean_cell"] / 1e8
    return selected


def panel_values(data: pd.DataFrame, field: str, zone: str) -> np.ndarray:
    indexed = data[data["bin"] == zone].set_index("comparison")
    return indexed.loc[COMPARISONS, field].to_numpy(dtype=float)


def draw(data: pd.DataFrame, output: Path, language: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    cn_font = FontProperties(family=["Arial", "Microsoft YaHei"], size=10.5)
    cn_tick_font = FontProperties(family=["Arial", "Microsoft YaHei"], size=9.5)
    x = np.arange(len(COMPARISONS), dtype=float)
    width = 0.32
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    fields = ["weight_change_pp", "p_1e8", "o_1e8"]

    if language == "cn":
        ylabels = [
            "权重变化（百分点）",
            "P贡献（10⁸ J/m²）",
            "O贡献（10⁸ J/m²）",
        ]
        xticklabels = ["1966—1995\n与1996—2025", "1982—2003\n与2004—2025"]
        legend_labels = ["20°N以南", "20°N及以北"]
    else:
        ylabels = [
            "Weight change (percentage points)",
            r"$P$ contribution ($10^8$ J m$^{-2}$)",
            r"$O$ contribution ($10^8$ J m$^{-2}$)",
        ]
        xticklabels = ["1966–1995\nvs 1996–2025", "1982–2003\nvs 2004–2025"]
        legend_labels = ["South of 20°N", "20°N and north"]

    handles = []
    for panel_index, (ax, field, ylabel) in enumerate(zip(axes, fields, ylabels)):
        for zone_index, zone in enumerate(ZONES):
            bars = ax.bar(
                x + (zone_index - 0.5) * width,
                panel_values(data, field, zone),
                width,
                color=COLORS[zone],
                edgecolor="white",
                linewidth=0.6,
                label=legend_labels[zone_index],
                zorder=3,
            )
            if panel_index == 0:
                handles.append(bars[0])
        ax.axhline(0.0, color="#666666", linewidth=0.8, zorder=2)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel(ylabel)
        if language == "cn":
            ax.yaxis.label.set_fontproperties(cn_font)
            for label in ax.get_xticklabels():
                label.set_fontproperties(cn_tick_font)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.01,
            0.98,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    legend_font = cn_tick_font if language == "cn" else FontProperties(family="Arial", size=9.5)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        prop=legend_font,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.84, wspace=0.36)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_flint_spec(data: pd.DataFrame, output: Path, source: Path) -> None:
    weight_rows = []
    contribution_rows = []
    comparison_labels = {
        COMPARISONS[0]: "1966–1995 vs 1996–2025",
        COMPARISONS[1]: "1982–2003 vs 2004–2025",
    }
    zone_labels = {
        "south_of_20N": "South of 20°N",
        "20N_and_north": "20°N and north",
    }
    for row in data.itertuples(index=False):
        weight_rows.append(
            {
                "comparison": comparison_labels[row.comparison],
                "latitude_zone": zone_labels[row.bin],
                "weight_change_pp": float(row.weight_change_pp),
            }
        )
        for component, value in (("P", row.p_1e8), ("O", row.o_1e8)):
            contribution_rows.append(
                {
                    "comparison": comparison_labels[row.comparison],
                    "latitude_zone": zone_labels[row.bin],
                    "component": component,
                    "contribution_1e8_j_m2": float(value),
                }
            )
    payload = {
        "source": {"path": str(source), "sha256": sha256(source)},
        "specs": {
            "weight_change": {
                "data": {"values": weight_rows},
                "semantic_types": {
                    "comparison": "Category",
                    "latitude_zone": "Category",
                    "weight_change_pp": "Number",
                },
                "chart_spec": {
                    "chartType": "Grouped Bar Chart",
                    "encodings": {
                        "x": {"field": "comparison"},
                        "y": {"field": "weight_change_pp"},
                        "group": {"field": "latitude_zone"},
                    },
                    "baseSize": {"width": 500, "height": 320},
                    "chartProperties": {"includeZero_y": True},
                },
            },
            "contributions": {
                "data": {"values": contribution_rows},
                "semantic_types": {
                    "comparison": "Category",
                    "latitude_zone": "Category",
                    "component": "Category",
                    "contribution_1e8_j_m2": "Number",
                },
                "chart_spec": {
                    "chartType": "Grouped Bar Chart",
                    "encodings": {
                        "x": {"field": "comparison"},
                        "y": {"field": "contribution_1e8_j_m2"},
                        "group": {"field": "latitude_zone"},
                        "column": {"field": "component"},
                    },
                    "baseSize": {"width": 760, "height": 320},
                    "chartProperties": {"includeZero_y": True, "independentYAxis": True},
                },
            },
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_and_validate(args.source, args.estimates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(data, args.output_dir / "FigS_ohc_latitude_contributions_en", "en")
    draw(data, args.output_dir / "FigS_ohc_latitude_contributions_cn", "cn")
    write_flint_spec(data, args.output_dir / "ohc_latitude_contributions_flint_spec.json", args.source)


if __name__ == "__main__":
    main()
