from pathlib import Path

from wnp_tc_analysis.src.publication_figure_map import (
    PUBLICATION_FIGURE_ALIASES,
    stage_publication_figures,
)


def test_v24_figure_map_has_unique_current_names():
    assert len(PUBLICATION_FIGURE_ALIASES) == 15
    assert len(set(PUBLICATION_FIGURE_ALIASES)) == 15
    assert "FigS01_agency_coverage_1945_2025" in PUBLICATION_FIGURE_ALIASES
    assert "FigS13_exclusive_coast_threshold_sensitivity" in PUBLICATION_FIGURE_ALIASES


def test_stage_uses_legacy_alias_without_changing_payload(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    old = source / "FigS10_agency_coverage_1945_2025.png"
    old.write_bytes(b"unchanged-figure-payload")

    staged = stage_publication_figures(source, destination)

    expected = destination / "FigS01_agency_coverage_1945_2025.png"
    assert staged == [expected]
    assert expected.read_bytes() == old.read_bytes()
