"""Canonical publication-figure names for manuscript release v2.4.

The v1.2.0 public release changes only display terminology, figure staging,
and the Figure 6 legend position.  The mapping preserves the v1.1.0 source
aliases so already-generated figure payloads can be staged without changing
their pixels or scientific content.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PUBLICATION_FIGURE_ALIASES = {
    "Fig04_genesis_conditional_track_field_decomposition": (
        "Fig04_genesis_propagation_decomposition",
    ),
    "Fig06_landfall_latitude_and_grouping": (
        "Fig06_landfall_latitude_and_grouping_v2",
    ),
    "FigS01_agency_coverage_1945_2025": (
        "FigS10_agency_coverage_1945_2025",
    ),
    "FigS02_early_year_projection": ("FigS11_early_year_projection",),
    "FigS03_start_year_sensitivity": ("FigS12_start_year_sensitivity",),
    "FigS04_activity_timeseries": ("FigS01_activity_timeseries",),
    "FigS05_lmi_location": ("FigS02_lmi_location",),
    "FigS06_path_definition_sensitivity": (
        "FigS05_path_definition_sensitivity",
        "FigS05_path_definition_sensitivity_v2",
    ),
    "FigS07_climate_mode_adjustment": (
        "FigS06_climate_mode_adjustment",
        "FigS06_climate_mode_adjustment_v2",
    ),
    "FigS08_wnpsh_fixed_contour_metrics": (
        "FigS09_wnpsh_fixed_contour_metrics",
    ),
    "FigS09_translation_speed_decomposition": (
        "FigS04_translation_speed_decomposition",
    ),
    "FigS10_ohc_latitude_contributions": (
        "FigS13_ohc_latitude_contributions",
    ),
    "FigS11_landfall_latitude_diagnostic": (
        "FigS08_landfall_latitude_diagnostic",
        "FigS08_landfall_latitude_diagnostic_v2",
    ),
    "FigS12_cutpoint_sensitivity": (
        "FigS07_cutpoint_sensitivity",
        "FigS07_cutpoint_sensitivity_v2",
    ),
    "FigS13_exclusive_coast_threshold_sensitivity": (
        "FigS03_exclusive_coast_threshold_sensitivity",
    ),
}


def stage_publication_figures(source: Path, destination: Path) -> list[Path]:
    """Stage PNG/PDF pairs under the current publication names."""
    staged: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for current, aliases in PUBLICATION_FIGURE_ALIASES.items():
        for suffix in (".png", ".pdf"):
            candidates = [source / f"{current}{suffix}"] + [
                source / f"{alias}{suffix}" for alias in aliases
            ]
            existing = next((path for path in candidates if path.exists()), None)
            if existing is None:
                continue
            output = destination / f"{current}{suffix}"
            shutil.copy2(existing, output)
            staged.append(output)
    return staged
