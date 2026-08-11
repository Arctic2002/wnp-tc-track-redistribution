# External data sources and redistribution boundary

## IBTrACS

- Product: International Best Track Archive for Climate Stewardship, Western Pacific, v04r01.
- DOI: https://doi.org/10.25921/82ty-9e16
- Use: USA composite and native USA, JMA/TOKYO, and CMA position and intensity fields, 1945–2025; primary analysis 1966–2025.
- The original CSV and processed result tables are not redistributed in this repository. Retrieval and analysis code is included; derived results are archived at https://doi.org/10.5281/zenodo.21879380.

## ERA5

- Pressure-level DOI: https://doi.org/10.24381/cds.bd0915c6
- Single-level DOI: https://doi.org/10.24381/cds.adbb2d47
- Use: monthly circulation and steering diagnostics over the western North Pacific.
- Large NetCDF source, intermediate fields, and derived tables are not redistributed in this repository. Retrieval and preprocessing code is included.

## ORAS5

- Product: ORAS5 global ocean reanalysis monthly data, ocean heat content for the upper 300 m.
- Product documentation: https://doi.org/10.5194/os-15-779-2019
- Use: monthly OHC300 sampled along agency-specific tropical-cyclone pathways, 1966–2025.
- Original annual archives, regional NetCDF subsets, matching tables, and decomposition results are not included in this repository. The OHC300 analysis and tests are included under `Verify/ohc_path_exposure/`; the regional subset and derived products are archived at https://doi.org/10.5281/zenodo.21879380.

## Climate indices

- ONI: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- PDO: https://www.ncei.noaa.gov/access/monitoring/pdo/
- The ONI and PDO values used in the analyses are archived with source attribution at https://doi.org/10.5281/zenodo.21879380.

## GSHHG

- Product: Global Self-consistent, Hierarchical, High-resolution Geography, v2.3.7.
- Dataset DOI: https://doi.org/10.5281/zenodo.7007502
- Method DOI: https://doi.org/10.1029/96JB00104
- Use: geometric shoreline intersections and mutually exclusive coast assignment.
- The original shapefiles and classified outputs are not included in this repository; retrieval and coast-assignment code is included.

## Maps-For-Free relief

- Source: https://maps-for-free.com/
- Use: cartographic relief shading in Figure 1 only; it is not an analytical input.
- The source tiles and study figures are not included in this repository.
