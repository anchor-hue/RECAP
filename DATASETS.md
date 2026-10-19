# Datasets

This document describes the seven reference–target benchmarks and two validation datasets used in RECAP. Original datasets were obtained from previously published studies, and the processed inputs used in this study are organized under `data/`.

## 1. Reference–Target Benchmarks

| Benchmark | Reference | Target | Data source |
|---|---|---|---|
| Murine N2–nanoPOTS | N2 | nanoPOTS | [MassIVE MSV000086809](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000086809); [MSV000084110](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000084110) |
| PBMC Batch1–Batch2 | PBMC Batch 1 | PBMC Batch 2 | [GEO GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) |
| nPOP plexDIA–TMT32 | plexDIA (Fig. 5) | TMTpro 32-plex (Fig. 7) | [MassIVE MSV000093494](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000093494); [MSV000094207](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000094207); [plexDIA data](https://doi.org/10.6084/m9.figshare.25762539.v1); [TMTpro data](https://doi.org/10.6084/m9.figshare.25277407.v1) |
| Human H1N1–IFNγ | H1N1Human | IFHuman | [FlowRepository FR-FCM-Z2NZ](https://flowrepository.org/id/FR-FCM-Z2NZ); [FR-FCM-Z2ZY](https://flowrepository.org/id/FR-FCM-Z2ZY) |
| Rhesus–Cynomolgus IFNγ | IFrhesus | IFcynomolgus | [FlowRepository FR-FCM-Z2ZY](https://flowrepository.org/id/FR-FCM-Z2ZY) |
| Derks QE–timsTOF | Q Exactive | timsTOF-SCP | [MassIVE MSV000089093](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000089093); [processed data](https://scp.slavovlab.net/Derks_et_al_2022) |
| Dou SCP–50-ng Boost | Processed single-cell profiles | 50-ng boosting profiles | [MassIVE MSV000084110](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000084110); [original publication and supplementary data](https://doi.org/10.1021/acs.analchem.9b03349) |

## 2. Synthetic Test Mixtures

For each benchmark, 1,000 test mixtures were generated from the processed target single-cell profiles.

For each mixture, latent cell-type proportions were sampled from a symmetric Dirichlet distribution with concentration parameter \(\alpha=0.3\). Cell counts were then sampled from a multinomial distribution with a total of 50 cells. The realized fractions, \(n_{sk}/50\), served as ground truth.

Cells were sampled with replacement within each cell type. The mean expression of sampled cells within each type defined the cell-type-specific expression ground truth. Mixture profiles were computed as fraction-weighted averages without additional normalization or noise; absent cell types were assigned zero-valued placeholders.

The same test mixtures were used across methods and random seeds. These test mixtures are distinct from the reference-derived pseudo-mixtures used for model training.

Processed inputs and ground-truth files are organized under `data/benchmarks/`. Their paths are specified in [`data/benchmark_manifest.json`](data/benchmark_manifest.json).

## 3. Experimental and Spatial Validation

### ProteoMixture

The ProteoMixture benchmark contains ten experimentally prepared physical mixtures with predefined component proportions and 6,129 proteins: five Tumor–Stroma mixtures and five Tumor–Lymphocyte mixtures. The original data are available through [PRIDE PXD044157](https://www.omicsdi.org/dataset/pride/PXD044157).

Three reference configurations were evaluated: all pure profiles, replicate holdout A, and replicate holdout B. The corresponding processed inputs are available under `data/proteomixture/`.

### SCPro

SCPro comprises spatial proteomic measurements of the mouse pancreatic tumor microenvironment, accompanied by flow-cytometry-sorted cell-type references. The data are available through ProteomeXchange/iProX under accessions PXD038882 and IPX0005555000 ([dataset record](https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD038882-2&test=no)).

Three reference panels were used: `immune9` for immune composition, `treg8` for T-cell states, and `main14` for consistency across reference configurations. Because independent ground-truth fractions and cell-type-specific expression profiles were unavailable for the spatial samples, this analysis focused on biological consistency rather than quantitative accuracy. Processed inputs are organized under `data/scpro/`.

## Data Usage

Please cite the original publications and data repositories when using these datasets. The repository provides processed benchmark inputs and analysis files to support reproducibility.
