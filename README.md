# Cell Type Coverage Analysis for Split-GAL4 Lines

Scripts to evaluate manually annotated matching hits and extrapolate the cell type
discovery curve to the complete set of split-GAL4 lines in the collection described in
[Meissner et al. 2025](https://elifesciences.org/reviewed-preprints/98405).

For more information, please see the corresponding paper: [Neuron types and distribution of a Drosophila driver line collection](https://www.biorxiv.org/content/10.64898/2026.10.06.756975v1).

## Annotated EM body ID-to-line matches

`data/all_body_id_annotations_*.csv` contain manually annotated matches between EM body IDs and split-GAL4 lines, including confirmed matches and highly ranked false matches from
the automated search.
Details on how the data were generated and annotated are provided in the [paper](https://www.biorxiv.org/content/10.64898/2026.10.06.756975v1). Note that the paper primarily focuses on matches between agglomerated cell types and split-GAL4 lines, whereas this resource covers individual EM body IDs.
These annotations are intended as a resource to facilitate future method development for automated matching. However, the dataset is not yet ready for use as an ML training dataset or benchmark: it lacks, for example, dedicated documentation (e.g., a datasheet), predefined training/validation/test splits, and an evaluation protocol.

## Annotated EM body ID to line matches

`data/all_body_id_annotations_*.csv` lists the manually annotated matches between EM body IDs and
split-GAL4 lines. It includes both confirmed matches and highly ranked false matches from
the automated search.

Note that it serves as a resource for evaluating matching methods, but it is not yet an AI- and benchmark-ready dataset, it still lacks, for example, detailed
documentation, predefined train/validation/test splits or an evaluation protocol.

## Installation

```bash
git clone https://github.com/Kainmueller-Lab/split_gal4_lines_coverage_analysis.git
cd split_gal4_lines_coverage_analysis
pip install -r requirements.txt
```

## Commands
### Run the coverage analysis

Computes discovery curves for the brain or VNC (or a specific region) and saves them
as pickle files.

```bash
python src/analyze_coverage.py \
    --cns_part brain \
    --config <path-to-config-file> \
    --output_folder <path-to-output-folder> \
    --skip_existing \
    --num_subsets 100 \
    --num_sample_obs 500 \
    --num_sample_opt 100 \
    --num_sample_ext 100
```

| Argument | Description |
|---|---|
| `--cns_part` | `brain` or `vnc` |
| `--config` | Path to the config file |
| `--output_folder` | Where the pickle files are written |
| `--skip_existing` | Skip regions that already have results saved as pickle |
| `--num_subsets` | Number of randomly sampled subsets from all annotated lines, used to create the corresponding number of models |
| `--num_sample_obs` | Number of random orderings used for Monte Carlo sampling to create the empirical discovery curve of annotated lines |
| `--num_sample_opt` | Number of discovery curves simulated during optimization for alpha |
| `--num_sample_ext` | Number of discovery curves simulated per model for final extrapolation |

### Visualize the results

Creates plots per region from the saved pickle files.

#### Per-region plots (brain)

```bash
python src/visualize_coverage.py \
    --input_folder <path-to-brain-output> \
    --cov_excel <path-to-brain-output>/coverages_brain.xlsx \
    --output_folder <path-to-plot-output-folder> \
    --cns_part brain \
    --do regions
```

#### Observed discovery curves (brain and VNC)

```bash
python src/visualize_coverage.py \
    --input_folder_brain <path-to-brain-output> \
    --input_folder_vnc <path-to-vnc-output> \
    --output_folder <path-to-plot-output-folder> \
    --cns_part both \
    --do dc_obs_both
```

| Argument | Description |
|---|---|
| `--input_folder` | Folder with the pickle files from step 3 (single CNS part) |
| `--input_folder_brain` | Same, for the brain (used with `--cns_part both`) |
| `--input_folder_vnc` | Same, for the VNC (used with `--cns_part both`) |
| `--cov_excel` | Excel file with coverage values per region |
| `--output_folder` | Where the plots are saved |
| `--cns_part` | `brain`, `vnc`, or `both` |
| `--do` | Plot type: `regions` (per-region plots), `dc_obs_both` (observed discovery curves for brain and VNC), `num_ct_per_line_both`(number of cell types per line for brain and VNC), `ct_freq_both` (cell type frequency for brain and vnc) |
