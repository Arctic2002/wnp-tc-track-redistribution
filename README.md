# Post-Genesis Track Redistribution in the Western North Pacific

This repository contains the analysis and figure-generation code associated with:

> Huang, D., and Y. Yue. *Post-Genesis Track Redistribution and Poleward Landfall-Latitude Shift in the Western North Pacific: Cross-Agency Evidence from 1966–2025.* Manuscript prepared for submission to *Journal of Geophysical Research: Oceans*.

The repository is the public development location for the software. The complete frozen reproduction package, including manuscript-facing derived data and the regional ORAS5 OHC300 subset, will be preserved separately in Zenodo under the reserved DOI [10.5281/zenodo.21879380](https://doi.org/10.5281/zenodo.21879380). The DOI will resolve after the Zenodo record is published.

## Repository contents

- `config/`: analysis domains, periods, thresholds, random seed, and environment specification.
- `core/`: shared download, preprocessing, track-table, and landfall-processing code.
- `paper2_dynamic/`: track, steering-flow, circulation, and translation-speed analyses.
- `wnp_tc_analysis/src/` and `wnp_tc_analysis/scripts/`: cross-agency, sensitivity, Shapley, circulation, landfall, and figure code.
- `wnp_tc_analysis/tests/` and `core/tests/`: tests for the shared and manuscript-specific analysis code.
- `Verify/`: independently structured verification code and method specifications for pathway closure, signed projection, counterfactual steering, native-stage landfall, and OHC300 exposure.
- `metadata/`: external-input registry and data-source notes.

Large source datasets, derived tables, NetCDF files, compressed decomposition products, and manuscript figures are intentionally excluded from GitHub. They are either available from their original providers or included, where redistribution is permitted, in the frozen Zenodo record.

## Environment

The recorded analysis environment uses Python 3.12 with the fixed random seed `202406`. Create the environment with:

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

Please use `CITATION.cff` when citing the software. The frozen Zenodo record has the reserved DOI [10.5281/zenodo.21879380](https://doi.org/10.5281/zenodo.21879380), which will resolve after publication.

The project code is licensed under the MIT License. External datasets retain their providers' terms.
