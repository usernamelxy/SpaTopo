# SpaTopo

SpaTopo: Spatial domain identification for spatially resolved transcriptomics via masked graph contrastive learning with adaptive routing.

SpaTopo combines gene expression and spatial coordinates by constructing a spatial neighbor graph, adaptively routing between block-masking and random-masking strategies based on spatial variability (Moran's I), and learning spot representations with a masked graph contrastive autoencoder. The learned embeddings are then clustered (mclust) to identify spatial domains.

## Repository Structure

```
SpaTopo/
├── SpaTopo/                          # Core package
│   ├── model.py                     # SpaTopo model (encoder, decoders, contrastive loss)
│   ├── graph_func.py                # Spatial neighbor graph construction
│   ├── clustering_func.py           # Clustering utilities (mclust / leiden / louvain)
│   └── utils_func.py                # Seed fixing, Moran's I routing strategy, etc.
├── SpaTopo_DLPFC_Clustering_rec.py   # Reproducible example on the DLPFC dataset
├── requirements.txt
└── README.md
```

## Installation

### 1. Python dependencies (Python >= 3.9 recommended)

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required (the code falls back to CPU).

### 2. R environment (required for mclust clustering)

The example uses the R package `mclust` through `rpy2`:

1. Install R (>= 4.0): https://cran.r-project.org/
2. Install mclust in R:

```r
install.packages("mclust")
```

3. Make sure `rpy2` can find your R installation (e.g. set the `R_HOME` environment variable on Windows).

> If you do not want to install R, you can replace `mclust_R(...)` in the example script with the `leiden(...)` function provided in `SpaTopo/clustering_func.py`.

## Data Preparation

The example reproduces results on the **DLPFC** dataset (12 human dorsolateral prefrontal cortex slices from the 10x Visium platform, Maynard et al., 2021), which is publicly available from the spatialLIBD project:

- Project page: http://research.libd.org/spatialLIBD/
- The data should be arranged in 10x Visium format (readable by `scanpy.read_visium`), plus a `metadata.tsv` containing the manual layer annotation (`layer_guess` column) for each spot.

Arrange the data as follows (each folder named by slice ID):

```
data/DLPFC/
├── 151507/
│   ├── filtered_feature_bc_matrix.h5
│   ├── spatial/
│   │   ├── tissue_positions_list.csv
│   │   └── ...
│   └── metadata.tsv          # with a 'layer_guess' column
├── 151508/
│   └── ...
└── ... (12 slices in total)
```

## Quick Start

Run the DLPFC example (processes all 12 slices by default):

```bash
python SpaTopo_DLPFC_Clustering_rec.py --data_root ./data/DLPFC
```

Run a single slice, e.g. 151507:

```bash
python SpaTopo_DLPFC_Clustering_rec.py --data_root ./data/DLPFC --sample_names 151507
```

Key arguments (all optional, defaults reproduce the paper setting):

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./data/DLPFC` | Root directory of the dataset |
| `--sample_names` | 12 DLPFC slices | Slice IDs to process |
| `--seed` | `42` | Random seed |
| `--n_top_genes` | `2000` | Number of highly variable genes |
| `--pca_n_comps` | `200` | PCA dimensions fed to the model |
| `--n_neighbors` | `12` | Neighbors in the spatial graph |
| `--epochs` | `200` | Training epochs |
| `--lr` | `0.01` | Learning rate |

## Output

For each slice, results are saved to `figures/recon/SpaTopo/<sample>/`:

- `SpaTopo.h5ad` — AnnData object with the learned embedding (`obsm['SpaTopo']`) and cluster labels (`obs['SpaTopo']`)
- `spatial_domain_<sample>.png` — manual annotation vs. SpaTopo clustering (with ARI)
- `loss_curve_<sample>.png` — training loss curve

The clustering quality is reported as ARI (Adjusted Rand Index) against the manual layer annotation, printed to the console at the end of each run.

## Citation

If you find SpaTopo useful in your research, please consider citing our work. (Citation information will be added upon publication.)

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
