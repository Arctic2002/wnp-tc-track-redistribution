"""Small deterministic tests for the independent pathway decomposition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "run_pathway_closure.py"
SPEC = importlib.util.spec_from_file_location("pathway_closure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_decomposition() -> None:
    rng = np.random.default_rng(17)
    years, families, height, width = 60, 3, 4, 5
    pi = rng.dirichlet(np.ones(families), size=years)
    conditional = rng.dirichlet(
        np.ones(height * width), size=(years, families)
    ).reshape(years, families, height, width)
    joint = pi[:, :, None, None] * conditional
    result = MODULE.decompose(pi, joint, np.arange(30), np.arange(30, 60))
    assert result["closure_max_abs"] < 1e-12
    assert abs(result["between_share"] + result["within_share"] - 1.0) < 1e-12


def test_block_indices() -> None:
    rng = np.random.default_rng(3)
    base = np.arange(30)
    draw = MODULE.block_indices(base, 3, rng)
    assert len(draw) == len(base)
    assert np.all((draw >= 0) & (draw < 30))
    assert all(
        np.array_equal(
            draw[index : index + 3], np.arange(draw[index], draw[index] + 3)
        )
        for index in range(0, len(draw), 3)
    )


def test_known_answer_between_only() -> None:
    pi = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    conditional = np.asarray(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 0.0]], [[0.0, 1.0]]],
        ]
    )
    joint = pi[:, :, None, None] * conditional
    result = MODULE.decompose(pi, joint, np.asarray([0]), np.asarray([1]))
    assert np.allclose(result["between"], result["delta"], atol=1e-12)
    assert np.allclose(result["within"], 0.0, atol=1e-12)
    assert abs(result["between_share"] - 1.0) < 1e-12
    assert abs(result["within_share"]) < 1e-12


def test_known_answer_within_only() -> None:
    pi = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    conditional = np.asarray(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[0.5, 0.5]], [[0.0, 1.0]]],
        ]
    )
    joint = pi[:, :, None, None] * conditional
    result = MODULE.decompose(pi, joint, np.asarray([0]), np.asarray([1]))
    assert np.allclose(result["between"], 0.0, atol=1e-12)
    assert np.allclose(result["within"], result["delta"], atol=1e-12)
    assert abs(result["between_share"]) < 1e-12
    assert abs(result["within_share"] - 1.0) < 1e-12


def test_density_normalization_and_relative_translation() -> None:
    paths = np.asarray(
        [
            [[100.0, 10.0], [102.0, 11.0], [104.0, 12.0]],
            [[120.0, 20.0], [122.0, 21.0], [124.0, 22.0]],
        ]
    )
    relative = paths - paths[:, :1, :]
    lon_edges = np.arange(-1.0, 7.0, 1.0)
    lat_edges = np.arange(-1.0, 4.0, 1.0)
    fields, valid = MODULE.storm_density_fields(
        relative, lon_edges, lat_edges
    )
    assert valid.all()
    assert np.allclose(fields.sum(axis=(1, 2)), 1.0)
    assert np.allclose(fields[0], fields[1])


def test_fixed_path_point_denominator_retains_domain_loss() -> None:
    paths = np.asarray(
        [
            [[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]],
            [[9.0, 9.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
        ]
    )
    edges = np.arange(-0.5, 2.5, 1.0)
    renormalized, renormalized_valid = MODULE.storm_density_fields(
        paths, edges, edges
    )
    fixed, fixed_valid = MODULE.storm_density_fields(
        paths,
        edges,
        edges,
        normalization="fixed_path_points",
    )
    assert np.array_equal(renormalized_valid, np.asarray([True, False]))
    assert fixed_valid.all()
    assert np.isclose(renormalized[0].sum(), 1.0)
    assert np.isclose(fixed[0].sum(), 0.5)
    assert np.isclose(fixed[1].sum(), 0.0)


def test_scenario_indices() -> None:
    years = np.arange(1966, 2026)
    early, late = MODULE.scenario_indices(years, 1982, 2004, 2025)
    assert np.array_equal(years[early], np.arange(1982, 2004))
    assert np.array_equal(years[late], np.arange(2004, 2026))


if __name__ == "__main__":
    test_exact_decomposition()
    test_block_indices()
    test_known_answer_between_only()
    test_known_answer_within_only()
    test_density_normalization_and_relative_translation()
    test_fixed_path_point_denominator_retains_domain_loss()
    test_scenario_indices()
    print(
        "PASS: exact closure, known-answer components, density geometry, "
        "block indexing, and period indexing"
    )
