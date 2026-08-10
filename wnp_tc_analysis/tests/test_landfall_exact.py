from shapely.geometry import LineString, box
from shapely.prepared import prep

from wnp_tc_analysis.src.landfall_latitude import geometry_points, sea_to_land_fractions


def test_single_sea_to_land_crossing():
    land = box(0.0, -1.0, 2.0, 1.0)
    line = LineString([(-2.0, 0.0), (1.0, 0.0)])
    fractions = sea_to_land_fractions(line, line.intersection(land.boundary), prep(land))
    assert len(fractions) == 1
    assert abs(line.interpolate(fractions[0], normalized=True).x) < 1e-9


def test_small_island_entry_and_exit_records_only_entry():
    land = box(-0.2, -0.2, 0.2, 0.2)
    line = LineString([(-1.0, 0.0), (1.0, 0.0)])
    fractions = sea_to_land_fractions(line, line.intersection(land.boundary), prep(land))
    assert len(fractions) == 1
    assert abs(line.interpolate(fractions[0], normalized=True).x + 0.2) < 1e-9


def test_land_to_sea_is_not_landfall():
    land = box(0.0, -1.0, 2.0, 1.0)
    line = LineString([(1.0, 0.0), (-1.0, 0.0)])
    fractions = sea_to_land_fractions(line, line.intersection(land.boundary), prep(land))
    assert fractions == []


def test_overlap_intersection_endpoints_are_extracted():
    points = geometry_points(LineString([(0.0, 0.0), (1.0, 0.0)]))
    assert len(points) == 2
