import os
import numpy as np
import pandas as pd
import pickle
import tomli
import logging

try:
    from plot_utils import plot_observed_discovery_curve
except ModuleNotFoundError:
    from src.plot_utils import plot_observed_discovery_curve

logger = logging.getLogger(__name__)


def load_pickle(pkl_fn):
    assert os.path.exists(pkl_fn), "Please check, pickle file does not exists! %s" % pkl_fn
    with open(pkl_fn, "rb") as inf:
        in_data = pickle.load(inf)
    return in_data


def save_dict_to_pickle(pkl_fn, out_dict):   
    with open(pkl_fn, "wb") as fout:
        pickle.dump(out_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)


def read_dict_from_pickle(input_folder, prefix, ann_set, region_str):
    if ann_set in ["both", "true"]:
        pkl_fn = os.path.join(input_folder, prefix + "_%s_%s.pkl" % (
            region_str.replace(" ","_"), "true"))
        assert os.path.exists(pkl_fn), "Please check, file %s does not exists!" % pkl_fn
        with open(pkl_fn, "rb") as inf:
            in_dict = pickle.load(inf)
    if ann_set in ["both", "true_maybe"]:
        pkl_fn = os.path.join(input_folder, prefix + "_%s_%s.pkl" % (
            region_str.replace(" ","_"), "true_maybe"))
        assert os.path.exists(pkl_fn), "Please check, file %s does not exists!" % pkl_fn
        with open(pkl_fn, "rb") as inf:
            in_dict_all = pickle.load(inf)
        if ann_set == "true_maybe":
            in_dict = in_dict_all
    if ann_set == "both":
        return in_dict, in_dict_all
    else:
        return in_dict, None


def load_config(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def get_cell_type_counts(cell_types, ann_cell_types, num_lines_per_ct):
    # get hits for each cell type in annotated lines
    all_ct_counts = {}
    for ct in cell_types:
        if ct in ann_cell_types:
            # if cell type is annotated, use the count
            all_ct_counts[ct] = int(num_lines_per_ct[ct])
        else:
            # if cell type is not annotated, use 0
            all_ct_counts[ct] = 0
    counts = np.array(list(all_ct_counts.values()))
    return counts, all_ct_counts


def get_correlation_table(ann_lines, cell_types, ct_per_line_subset):
    # get table with cell types of region in columns and lines in rows
    df_corr = pd.DataFrame(
            data=np.zeros((len(ann_lines), len(cell_types)), dtype=np.uint8),
            index=ann_lines,
            columns=cell_types
            )
    # fill correlation table: which cell types in which line
    for i, line in enumerate(ann_lines):
        df_corr.loc[line, ct_per_line_subset[line]] = 1
    y_obs = df_corr.to_numpy()
    return y_obs


def estimate_mean_discovery_curve(y_obs, n_trials=1000):
    """
    Monte Carlo estimation of the average discovery curve
    y_obs: binary matrix (lines x cell types), 1 if cell type is observed
    n_trials: number of random orderings
    """
    L, C = y_obs.shape
    discovery_curves = np.zeros((n_trials, L))
    for trial in range(n_trials):
        order = np.random.permutation(L)
        seen = np.zeros(C, dtype=bool)
        unique_count = 0
        for i, line_idx in enumerate(order):
            new = np.logical_and(y_obs[line_idx], ~seen)
            unique_count += np.sum(new)
            seen = np.logical_or(seen, y_obs[line_idx])
            discovery_curves[trial, i] = unique_count

    mean_curve = np.mean(discovery_curves, axis=0)
    return mean_curve, discovery_curves


def get_discovery_curves(y_obs, cell_types, region_str, ann_str,
        num_sample_obs=1000, output_folder="", plot=False, save_pickle=False,
        skip_existing=False, prefix="observed"):
    """
    Compute and optionally plot the mean discovery curve for annotated cell types across lines.

    Parameters:
        y_obs (np.ndarray): Binary matrix (lines x cell types), 1 if cell type is observed in line.
        cell_types (list): List of cell type names.
        region_str (str): Region identifier (used for filenames).
        ann_str (str): Annotation set identifier (used for filenames).
        num_sample_obs (int): Number of Monte Carlo samples for estimating the curve.
        output_folder (str): Directory to save output files (plots, pickles).
        plot (bool): If True, save a plot of the discovery curve.
        save_pickle (bool): If True, save the discovery curves to a pickle file.
        skip_existing (bool): If True and pickle exists, load instead of recomputing.
        prefix (str): Prefix for output filenames.

    Returns:
        mean_dc (np.ndarray): Mean discovery curve (averaged over samples).
        dcs (np.ndarray): All sampled discovery curves (n_trials x n_lines).
    """
    pkl_fn = os.path.join(output_folder, "%s_%s_%s.pkl" % (
        prefix, region_str.replace(" ","_"), ann_str))

    # if pickle file already exists, load data
    if skip_existing and os.path.exists(pkl_fn):
        logger.info("Skipping MC sampling for observed data for %s in %s" % (
            region_str, ann_str))
        in_dict = load_pickle(pkl_fn)
        dcs = in_dict["discovery_curves"]
        mean_dc = np.mean(dcs, axis=0)
    else:
        # get mean discovery curve by sampling
        mean_dc, dcs = estimate_mean_discovery_curve(y_obs, n_trials=num_sample_obs)
        
        # save to pickle
        if save_pickle:
            out_dict = {
                    "num_cell_types": len(cell_types),
                    "discovery_curves": dcs
                    }
            with open(pkl_fn, "wb") as fout:
                pickle.dump(out_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)
        
    # plot observed data
    if plot:
        lower_curve_obs = np.percentile(dcs, 2.5, axis=0)
        upper_curve_obs = np.percentile(dcs, 97.5, axis=0)
        plot_fn = os.path.join(output_folder,
                "%s_discovery_curve_%s_%s.png" % (prefix, region_str.replace(" ", "_"), ann_str))
        plot_observed_discovery_curve(
                list(np.arange(1, len(mean_dc) + 1)),
                mean_dc, lower_curve_obs, upper_curve_obs, len(cell_types),
                region_str, ann_str, plot_fn
                )
    return mean_dc, dcs


def estimate_probabilities(counts, alpha=1.0):
    """
    Estimate a probability distribution with Laplace smoothing.

    Parameters:
        counts (list): Observed counts for each color.
        alpha (float): Smoothing parameter.

    Returns:
        list: Estimated probabilities.
    """
    M = len(counts)
    N = sum(counts)
    return [(c + alpha) / (N + M * alpha) for c in counts]

