# Post-Genesis Track Redistribution in the Western North Pacific

This repository provides the analysis and figure-generation code for the companion manuscript:

> Huang, D., and Y. Yue. *Post-Genesis Track Redistribution and Poleward Landfall-Latitude Shift in the Western North Pacific: Cross-Agency Evidence from 1966–2025.*

The associated derived data and regional monthly ORAS5 OHC300 subset are archived on Zenodo at [10.5281/zenodo.22011203](https://doi.org/10.5281/zenodo.22011203).

## Repository contents

- `config/`: analysis domains, periods, thresholds, random seed, and environment specification.
- `core/`: shared download, preprocessing, track-table, and landfall-processing code.
- `paper2_dynamic/`: track, steering-flow, circulation, and translation-speed analyses.
- `wnp_tc_analysis/src/` and `wnp_tc_analysis/scripts/`: cross-agency, sensitivity, Shapley, circulation, landfall, and figure code.
- `wnp_tc_analysis/tests/` and `core/tests/`: tests for the shared and manuscript-specific analysis code.
- `Verify/`: independently structured verification code and method specifications for pathway closure, signed projection, counterfactual steering, native-stage landfall, and OHC300 exposure.
- `metadata/`: external-input registry and data-source notes.

Original third-party datasets remain available from their providers. Redistributable derived tables, figures, and the regional ORAS5 OHC300 subset are included in the Zenodo record.

## Environment

The analysis environment uses Python 3.12 with the fixed random seed `202406`. Create the environment with:

```bash
conda env create -f config/environment.yml
conda activate tc-wnp
```

The core validation suite can then be run from the repository root:

```bash
python -m pytest core/tests wnp_tc_analysis/tests \
  Verify/ohc_path_exposure/tests Verify/pathway_closure/tests
```

Full regeneration requires the external inputs listed in `metadata/external_inputs.csv`. No credentials or API keys are included. Copernicus users must configure their own CDS credentials.

## Data sources

The analysis uses IBTrACS v04r01, ERA5, ORAS5, ONI, PDO, and GSHHG. Source locations, identifiers, roles, and redistribution boundaries are documented in `metadata/DATA_SOURCES.md` and `metadata/external_inputs.csv`.

## Citation and license

Please use `CITATION.cff` when citing the software and associated data package. Its DOI is [10.5281/zenodo.22011203](https://doi.org/10.5281/zenodo.22011203).

The project code is licensed under the MIT License. External datasets retain their providers' terms.
