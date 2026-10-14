import os
import numpy as np
import pickle
import random
from scipy.optimize import minimize_scalar
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import logging
import time
import concurrent.futures
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from utils import (
        get_correlation_table,
        get_cell_type_counts,
        get_discovery_curves,
        estimate_probabilities,
        load_pickle,
        save_dict_to_pickle,
    )
    from plot_utils import (
        plot_observed_and_extrapolated_discovery_curves,
        plot_lines_from_dict,
    )
except ModuleNotFoundError:
    from src.utils import (
        get_correlation_table,
        get_cell_type_counts,
        get_discovery_curves,
        estimate_probabilities,
        load_pickle,
        save_dict_to_pickle,
    )
    from src.plot_utils import (
        plot_observed_and_extrapolated_discovery_curves,
        plot_lines_from_dict,
    )


logger = logging.getLogger(__name__)


def get_line_and_ct_info_for_subset(cell_types, ann_lines, ct_per_line):
    # update ct_per_line
    ct_per_line_subset = {
        line: list(cell_types[np.isin(cell_types, ct_per_line[line])])
        for line in ann_lines
    }

    # get number of cell types per line specific to region / subset
    num_ct_per_line_subset = {
        line: np.sum(np.isin(cell_types, ct_per_line[line])) for line in ann_lines
    }

    y_obs_subset = get_correlation_table(ann_lines, cell_types, ct_per_line_subset)

    return ct_per_line_subset, num_ct_per_line_subset, y_obs_subset


# Sample one line (without replacement)
def sample_line(num_cell_types, mu_values, probs):
    mu_sample = np.random.choice(mu_values)  # draw from empirical distribution
    if mu_sample == 0:
        return np.array([], dtype=int)
    return np.random.choice(
        num_cell_types, size=min(mu_sample, num_cell_types), replace=False, p=probs
    )


def simulate_discovery_vectorized(
    num_cell_types, p_smoothed, total_lines, num_ct_per_line, n_reps=1000
):
    discovery_curves = np.zeros((n_reps, total_lines), dtype=int)
    for rep in range(n_reps):
        discovered = np.zeros(num_cell_types, dtype=bool)
        mu_samples = np.random.choice(num_ct_per_line, size=total_lines)
        sampled_indices = np.array(
            [
                np.random.choice(num_cell_types, size=mu, replace=False, p=p_smoothed)
                if mu > 0
                else np.array([], dtype=int)
                for mu in mu_samples
            ],
            dtype=object,
        )
        for t in range(total_lines):
            if len(sampled_indices[t]) > 0:
                discovered[sampled_indices[t]] = True
            discovery_curves[rep, t] = np.sum(discovered)
    return discovery_curves


def make_weights(length, mode="quadratic", power=2.0, gamma=4.0):
    """
    Returns weights w of shape (length,) increasing with t.
    - mode="linear":     w_t ~ t
    - mode="quadratic":  w_t ~ t^power (default power=2)
    - mode="exp":        w_t ~ exp(gamma * x), x in [0,1]
    """
    t = np.arange(length)
    x = (t + 1) / (length)  # range (0,1]
    if mode == "linear":
        w = x
    elif mode == "quadratic":
        w = x**power
    elif mode == "exp":
        w = np.exp(gamma * (x - 1.0))   # small at start, ~1 at end if gamma>0
    else:
        raise ValueError("Unknown mode")
    return w / w.sum()  # normalize for numerical stability


def optimize_simulation(
    counts, num_ct_per_line, target_curve, num_sample_opt=1000,
    num_iter_opt=1000, mean_curve_obs_subset=None, output_folder=None
):
    """
    Optimize the smoothing parameter alpha for cell type discovery simulation.

    Parameters:
        counts (array-like): Observed counts for each cell type.
        num_ct_per_line (array-like): Number of cell types per line in the subset.
        target_curve (array-like): The observed discovery curve to fit.
        num_sample_opt (int): Number of Monte Carlo samples in objective per optimization step.
        num_iter_opt (int): Number of iterations (optimization steps) for the optimization process.

    Returns:
        float: The optimal alpha value minimizing the mean squared error between simulated and observed curves.
    """

    #@use_named_args([Real(1e-5, 10.0, name="alpha")])
    
    def objective(alpha):
        p_smoothed = estimate_probabilities(counts, alpha)
        # Run your discovery simulation with current p_i
        simulated_curves = simulate_discovery_vectorized(
            num_cell_types=len(counts),
            p_smoothed=p_smoothed,
            total_lines=len(target_curve),
            num_ct_per_line=num_ct_per_line,
            n_reps=num_sample_opt,
        )
        mean_simulated = np.mean(simulated_curves, axis=0)
        # Calculate error
        w = make_weights(len(mean_simulated), mode="linear")
        error = np.average((mean_simulated - target_curve) ** 2, weights=w)
        return error

    # plot loss
    #alphas = np.linspace(0, 2, 100)   # 200 evenly spaced points
    #loss_values = np.array([objective(a) for a in alphas])
    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    #plt.clf()
    #plt.figure(figsize=(8, 5))
    #plt.plot(alphas, loss_values, marker='', linewidth=2)
    #plt.xlabel("alpha")
    #plt.ylabel("loss / error")
    #plt.title("Loss function vs. alpha (0 → 2)")
    #plt.grid(True)
    #outfn = os.path.join(output_folder, f"loss_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    #plt.savefig(outfn)
    
    #result = gp_minimize(
    #    func=objective,
    #    dimensions=[Real(0.0, 1.0, name="alpha")],
    #    x0=[0.1],
    #    n_calls=num_iter_opt,
    #    random_state=42,
    #)
    result = minimize_scalar(
            objective,
            bounds=(0.0, 1.0),
            method="bounded"
    )
    logger.info(f"Best alpha: {result.x}")
    logger.info(f"Minimum loss: {result.fun}")
    print(result.success, result.status, result.message)
    print("-------------------------------------------")
    return result.x, result.fun


def get_simulated_discovery_curves(
    num_cell_types,
    p_cell_types,
    total_lines,
    num_ct_per_line,
    region_str,
    ann_set_name,
    alpha,
    num_sample_ext=1000,
    output_folder="",
    save_pickle=False,
    skip_existing=False,
):
    pkl_fn = os.path.join(
        output_folder,
        "simulated_%s_%s_a_%s.pkl"
        % (region_str.replace(" ", "_"), ann_set_name, str(alpha).replace(".", "_")),
    )

    if skip_existing and os.path.exists(pkl_fn):
        logger.info("Skipping simulation for %s in %s" % (region_str, ann_set_name))
        in_dict = load_pickle(pkl_fn)
        sim_dc = in_dict["discovery_curves"]
        logger.debug("loaded simulated data: ", sim_dc.shape)
        mean_dc = np.mean(sim_dc, axis=0)
        return mean_dc, sim_dc

    # get simulated discovery curves
    discovery_curves = simulate_discovery_vectorized(
        num_cell_types=num_cell_types,
        p_smoothed=p_cell_types,
        total_lines=total_lines,
        num_ct_per_line=num_ct_per_line,
        n_reps=num_sample_ext,
    )
    mean_dc = np.mean(discovery_curves, axis=0)

    # save to pickle
    if save_pickle:
        out_dict = {
            "p": p_cell_types,
            "num_cell_types": num_cell_types,
            "num_ct_per_line": num_ct_per_line,
            "discovery_curves": discovery_curves,
            "alpha": alpha,
        }
        save_dict_to_pickle(pkl_fn, out_dict)

    logger.debug(discovery_curves.shape)

    return mean_dc, discovery_curves


def _simulate_subset_worker(args_tuple):
    (
        subset,
        subset_size,
        ann_lines,
        cell_types,
        ct_per_line,
        num_lines_per_ct,
        mean_target_curve,
        region_str,
        ann_set_name,
        num_sample_obs,
        num_sample_opt,
        num_iter_opt,
        num_sample_ext,
        lines_total,
        output_folder,
        testset,
        testset_size,
        output_folder
    ) = args_tuple
    
    np.random.seed()  # ensure randomness in each process
    
    # train split
    if testset is not None:
        # take fixed testset
        trainset = np.array(ann_lines)[np.isin(ann_lines, testset, invert=True)]
        subset_size = 1 - float(len(testset)) / len(trainset)
    else:
        # take random train and test set
        train_part = 1 - testset_size
        trainset_size = int(len(ann_lines) * train_part)
        trainset = np.random.choice(ann_lines, trainset_size, replace=False)
        subset_size = 1 - float(len(ann_lines)-len(trainset)) / len(trainset)
    
    subset_size = int(len(trainset) * subset_size)
    logger.info("taking training subset of size: %i" % subset_size)
    #logger.info("ann lines: %i, trainset: %i, subset: %i") % (
    #        len(ann_lines), trainset_size, subset_size)
    
    ct_per_line_trainset, num_ct_per_line_trainset, y_obs_trainset = (
        get_line_and_ct_info_for_subset(cell_types, trainset, ct_per_line)
    )
    # get target discovery curve observed for complete train set
    mean_curve_obs, _ = get_discovery_curves(
        y_obs_trainset,
        cell_types,
        region_str,
        ann_set_name,
        num_sample_obs=num_sample_obs,
        plot=False,
        save_pickle=False,
        skip_existing=False,
    )

    # sample subset from train set to get p_count, p^alpha_cell_type
    ann_lines_subset = np.random.choice(trainset, subset_size, replace=False)
    ct_per_line_subset, num_ct_per_line_subset, y_obs_subset = (
        get_line_and_ct_info_for_subset(cell_types, ann_lines_subset, ct_per_line)
    )
    mean_curve_obs_subset, _ = get_discovery_curves(
        y_obs_subset,
        cell_types,
        region_str,
        ann_set_name,
        num_sample_obs=num_sample_obs,
        plot=False,
        save_pickle=False,
        skip_existing=False,
    )
    # get cell type counts for subset
    ann_cts_subset = np.unique(np.concatenate(list(ct_per_line_subset.values())))
    num_lines_per_ct_subset = {
        ct: sum([int(ct in ct_per_line_subset[line]) for line in ann_lines_subset])
        for ct in cell_types
    }
    counts_subset, _ = get_cell_type_counts(cell_types, ann_cts_subset, num_lines_per_ct_subset)
    
    # get alpha, fit simulated curve to observed curve
    alpha, val = optimize_simulation(
            counts_subset,
            list(num_ct_per_line_subset.values()),
            mean_curve_obs, #target curve
            num_sample_opt=num_sample_opt,
            num_iter_opt=num_iter_opt,
            mean_curve_obs_subset=mean_curve_obs_subset,
            output_folder=output_folder
    )
    sim_dc_mean_subset, sim_dc_subset = None, None
    diff = None 
    if alpha is not None: # and np.abs(val) < 50:
        p_cell_type = estimate_probabilities(counts_subset, alpha=float(alpha))
        logger.info("simulate dc with MC trials of %i" % num_sample_ext)
        sim_dc_mean_subset, sim_dc_subset = get_simulated_discovery_curves(
            len(cell_types),
            p_cell_type,
            lines_total,
            list(num_ct_per_line_subset.values()),
            region_str,
            ann_set_name,
            alpha=float(alpha),
            num_sample_ext=num_sample_ext,
            output_folder=output_folder,
            save_pickle=False,
            skip_existing=False,
        )
        diff = (sim_dc_mean_subset[len(mean_target_curve)-1]-mean_target_curve[-1])
    return (subset, mean_curve_obs, alpha, sim_dc_mean_subset, sim_dc_subset,
            trainset_size, subset_size, diff, mean_curve_obs_subset)


def simulate(
    ann_lines,
    ann_cell_types,
    cell_types,
    ct_per_line,
    num_lines_per_ct,
    lines_total,
    cns_part,
    region_key,
    ann_set_name,
    subset_size=0.5,
    num_subsets=100,
    num_sample_obs=1000,
    num_sample_opt=1000,
    num_sample_ext=1000,
    num_iter_opt=1000,
    output_folder="",
    skip_existing=False,
    num_workers=None,
):
    # ---------------------------------------------------------------------------
    # get discovery curve for all annotations --> target curve
    if type(cell_types) is not np.array:
        cell_types = np.array(cell_types)
    ct_per_line_all, num_ct_per_line_all, y_obs_all = get_line_and_ct_info_for_subset(
        cell_types, ann_lines, ct_per_line
    )

    if region_key == "MANC":
        region_str = "MANC"
    elif region_key == "hemibrain":
        region_str = "Hemibrain"
    else:
        region_str = " ".join([cns_part, region_key])
    logger.info(
        "Get observed discovery rate for region: %s, annotation set: %s, and num MC samples: %i, and skip existing: %s"
        % (region_str, ann_set_name, num_sample_obs, str(skip_existing))
    )
    start_time = time.time()
    mean_obs_dc_all, dcs_obs_all = get_discovery_curves(
        y_obs_all,
        cell_types,
        region_str,
        ann_set_name,
        num_sample_obs=num_sample_obs,
        output_folder=output_folder,
        plot=True,
        save_pickle=True,
        skip_existing=True,
        prefix="observed_all_%i" % num_sample_obs,
    )
    elapsed = time.time() - start_time
    logger.info(f"get discovery curves for all annotations took {elapsed:.2f} seconds")
    x_obs_dc = list(np.arange(1, len(mean_obs_dc_all) + 1))
    lower_obs_dc_all = np.percentile(dcs_obs_all, 2.5, axis=0)
    upper_obs_dc_all = np.percentile(dcs_obs_all, 97.5, axis=0)
    num_ct_annotated = len(np.unique(np.concatenate(list(ct_per_line_all.values()))))

    # ---------------------------------------------------------------------------
    # get discovery curve for subset of annotations and optimize alpha
    x_obs_subsets = {}
    dc_obs_subsets = {}
    dcs_ext_subsets = {}
    mean_dc_ext_subsets = {}
    mean_dc_pbase_subsets = {}
    alphas = []
    trainset_size = 0
    subset_size = 0
    testset_diffs = []
    rmse_mean = None
    rmse_std = None

    # check if extrapolated discovery curves already exist
    pkl_ext_fn = os.path.join(
        output_folder,
        "extrapolated_%s_%s_subsets_%s_sample_ext_%s.pkl"
        % (
            region_str.replace(" ", "_"),
            ann_set_name,
            str(num_subsets),
            str(num_sample_ext),
        ),
    )
    # load discovery curves from file if exists and flag is set
    if skip_existing and os.path.exists(pkl_ext_fn):
        logger.info("Skipping simulation for %s in %s" % (region_str, ann_set_name))
        in_dict = load_pickle(pkl_ext_fn)
        all_ext_dcs = in_dict["dc_ext_all"]
        #dc_ext_mean_per_model = in_dict["dc_ext_mean_per_model"]
        testset_diffs = in_dict["diff"]
    else:
        # sample different subsets
        #testset_size = int(len(ann_lines) * 0.3)
        testset_size = 0.25
        #testset = np.random.choice(ann_lines, testset_size, replace=False)
        testset = None
        #if subset_size < 1:
        #    subset_size = int(len(ann_lines) * subset_size)
        #logger.info("Sample %i lines out of %i" % (subset_size, len(ann_lines)))
        start_time_opt_all_subsets = time.time()
        subset_args = [
            (
                subset,
                subset_size,
                ann_lines,
                cell_types,
                ct_per_line,
                num_lines_per_ct,
                mean_obs_dc_all,
                region_str,
                ann_set_name,
                num_sample_obs,
                num_sample_opt,
                num_iter_opt,
                num_sample_ext,
                lines_total,
                output_folder,
                testset,
                testset_size,
                output_folder
            )
            for subset in range(num_subsets)
        ]
        if num_workers is None or num_workers == 1:
            results = [_simulate_subset_worker(args) for args in subset_args]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                results = list(executor.map(_simulate_subset_worker, subset_args))
        for subset, mean_curve_obs, ret, sim_dc_mean_subset, sim_dc_subset, trainset_samples, \
                subset_samples, diff, mean_curve_obs_pbase in results:
            if ret is not None:
                dc_obs_subsets[subset] = mean_curve_obs
                x_obs_subsets[subset] = list(np.arange(1, len(mean_curve_obs) + 1))
                alphas.append(ret)
                dcs_ext_subsets[subset] = sim_dc_subset
                mean_dc_ext_subsets[subset] = sim_dc_mean_subset
                trainset_size = trainset_samples
                subset_size = subset_samples
                testset_diffs.append(diff)
                mean_dc_pbase_subsets[subset] = mean_curve_obs_pbase
        elapsed = time.time() - start_time_opt_all_subsets
        logger.info(f"Completed simulation for all subsets in {elapsed:.2f} seconds")

        # get mean and confidence interval of extrapolated discovery curves
        all_ext_dcs = np.concatenate(list(dcs_ext_subsets.values()), axis=0)
        all_ext_dcs_mean = np.concatenate(list(mean_dc_ext_subsets.values()), axis=0)
        all_obs_dcs_mean = np.concatenate(list(dc_obs_subsets.values()), axis=0)
        testset_diffs = np.array(testset_diffs)
            
        # save pickle with extrapolated discovery curves
        pkl_ext_fn = os.path.join(
            output_folder,
            "extrapolated_%s_%s_subsets_%s_sample_ext_%s.pkl"
            % (
                region_str.replace(" ", "_"),
                ann_set_name,
                str(num_subsets),
                str(num_sample_ext),
            ),
        )
        out_dict = {
            "num_cell_types": len(cell_types),
            "dc_ext_all": all_ext_dcs,
            "dc_ext_mean_per_model": all_ext_dcs_mean,
            "dc_obs_mean_trainsets": all_obs_dcs_mean,
            "diff": testset_diffs,
            "alphas": alphas,
        }
        save_dict_to_pickle(pkl_ext_fn, out_dict)

        # plot all subsets
        xdata_all_subsets = {}
        ydata_all_subsets = {}
        color_all_subsets = {}
        linestyle_all_subsets = {}
        for subset in range(num_subsets):
            if subset not in list(x_obs_subsets.keys()):
                continue
            # add extrapolated discovery curves
            if subset in mean_dc_ext_subsets:
                xdata_all_subsets["ext_%d" % subset] = list(
                    np.arange(1, len(mean_dc_ext_subsets[subset]) + 1)
                )
                ydata_all_subsets["ext_%d" % subset] = mean_dc_ext_subsets[subset]
                color_all_subsets["ext_%d" % subset] = "C%d" % (subset % 10)
                linestyle_all_subsets["ext_%d" % subset] = ":"
            
            # add observed train
            xdata_all_subsets[subset] = x_obs_subsets[subset]
            ydata_all_subsets[subset] = dc_obs_subsets[subset]
            color_all_subsets[subset] = "C%d" % (subset % 10)
            linestyle_all_subsets[subset] = "--"
            
            # add observed train subset
            xdata_all_subsets["sub_%d" % subset] = list(np.arange(1, len(
                mean_dc_pbase_subsets[subset]) + 1))
            ydata_all_subsets["sub_%d" % subset] = mean_dc_pbase_subsets[subset]
            color_all_subsets["sub_%d" % subset] = "C%d" % (subset % 10)
            linestyle_all_subsets["sub_%d" % subset] = "-"
        
        plot_fn = os.path.join(
            output_folder, "plot_all_subsets_%s_%s.png" % (region_str, ann_set_name)
        )
        plot_lines_from_dict(
            xdata_all_subsets,
            ydata_all_subsets,
            color_dict=color_all_subsets,
            linestyle_dict=linestyle_all_subsets,
            num_ct_annotated=num_ct_annotated,
            num_lines_annotated=len(list(num_ct_per_line_all.keys())),
            output_filename=plot_fn,
        )
        # calculate error
        diff_mean = np.mean(testset_diffs)
        diff_std = np.std(testset_diffs)
        mse_mean = np.mean(np.array(testset_diffs)**2)
        mse_std = np.std(np.array(testset_diffs)**2)
        rmse_mean = np.mean(np.sqrt(np.array(testset_diffs)**2))
        rmse_std = np.std(np.sqrt(np.array(testset_diffs)**2))
        diff_min = np.min(np.abs(testset_diffs))
        diff_max = np.max(np.abs(testset_diffs))
        logger.info(f"Diff min: {diff_min:.2f}, diff max: {diff_max:.2f}")
        logger.info(f"Diff {diff_mean:.2f} +/- {diff_std:.2f}, MSE {mse_mean:.2f} +/- {mse_std:.2f}, RMSE {rmse_mean:.2f} +/- {rmse_std:.2f}")
        
        # plot all subsets different
        xdata_all_subsets = {}
        ydata_all_subsets = {}
        color_all_subsets = {}
        linestyle_all_subsets = {}
        for subset in range(num_subsets):
            if subset not in list(x_obs_subsets.keys()):
                continue
            xdata_all_subsets[subset] = x_obs_subsets[subset]
            ydata_all_subsets[subset] = dc_obs_subsets[subset]
            #color_all_subsets[subset] = "C%d" % (subset % 10)
            color_all_subsets[subset] = "moccasin"
            linestyle_all_subsets[subset] = "-"
            # add extrapolated discovery curves
            if subset in mean_dc_ext_subsets:
                if subset == 0:
                    k = "model_example"
                else:
                    k = "ext_%d" % subset
                xdata_all_subsets[k] = list(
                    np.arange(1, len(mean_dc_ext_subsets[subset]) + 1)
                )
                ydata_all_subsets[k] = mean_dc_ext_subsets[subset]
                color_all_subsets[k] = "moccasin"
                linestyle_all_subsets[k] = "--"
        # add observed dc for all annotated lines
        xdata_all_subsets["obs"] = list(np.arange(1, len(mean_obs_dc_all) + 1))
        ydata_all_subsets["obs"] = mean_obs_dc_all
        color_all_subsets["obs"] = "blue"
        linestyle_all_subsets["obs"] = "-"
        
        # create filename
        plot_fn = os.path.join(
            output_folder, "plot_all_subsets_%s_%s_2.png" % (region_str, ann_set_name)
        )
        plot_lines_from_dict(
            xdata_all_subsets,
            ydata_all_subsets,
            color_dict=color_all_subsets,
            linestyle_dict=linestyle_all_subsets,
            num_ct_annotated=None,
            num_lines_annotated=len(list(num_ct_per_line_all.keys())),
            output_filename=plot_fn,
            subset_size=subset_size,
            trainset_size=trainset_size,
            testset_error_mean=rmse_mean,
            testset_error_std=rmse_std
        )

    #if len(testset_diffs) == 0:
    #    for subset in range(num_subsets):
    #        num_dcs = int(all_ext_dcs.shape[0] / num_subsets)
    #        dcs_subset = all_ext_dcs[subset*num_dcs:subset*num_dcs+num_dcs, :]
    #        mean_dc_ext = np.mean(dcs_subset, axis=0)
    #        diff = (mean_dc_ext[len(mean_obs_dc_all)-1]-mean_obs_dc_all[-1])
    #        testset_diffs.append(diff)
    if rmse_mean is None or rmse_std is None:
        rmse_mean = np.mean(np.sqrt(np.array(testset_diffs)**2))
        rmse_std = np.std(np.sqrt(np.array(testset_diffs)**2))
        logger.info(f"RMSE {rmse_mean:.2f} +/- {rmse_std:.2f}")

    # extrapolate error
    chunk_size = num_subsets
    coeffs = []
    sign_dists = []
    for i, diff in enumerate(testset_diffs):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        dc_chunk = all_ext_dcs[start:end, :]

        obs_x1 = mean_obs_dc_all[-1]
        sim_x1 = dc_chunk[:, len(mean_obs_dc_all)-1]
        mean_sim_x1 = np.mean(sim_x1)
        std_sim_x1 = np.std(sim_x1)
        coeff = (mean_sim_x1 - obs_x1) / std_sim_x1
        coeffs.append(coeff)
        sim_x2 = dc_chunk[:, -1]
        std_sim_x2 = np.std(sim_x2)
        sign_dist = coeff * std_sim_x2
        sign_dists.append(sign_dist)
    rmse_ext_mean = np.mean(np.sqrt(np.array(sign_dists)**2))
    rmse_ext_std = np.std(np.sqrt(np.array(sign_dists)**2))

    mean_ext_dc = np.mean(all_ext_dcs, axis=0)
    x_ext_dc = np.array(list(np.arange(1, len(mean_ext_dc) + 1)), dtype=int)
    lower_ext_dc = np.percentile(all_ext_dcs, 2.5, axis=0)
    upper_ext_dc = np.percentile(all_ext_dcs, 97.5, axis=0)
    num_ct_extrapolated = float(mean_ext_dc[x_ext_dc == lines_total])


    # plot extrapolated discovery curves
    plot_fn = os.path.join(
        output_folder,
        "extrapolated_discovery_curve_%s_%s.svg" % (region_str, ann_set_name),
    )
    plot_observed_and_extrapolated_discovery_curves(
        x_obs_dc,
        mean_obs_dc_all,
        lower_obs_dc_all,
        upper_obs_dc_all,
        x_ext_dc,
        mean_ext_dc,
        lower_ext_dc,
        upper_ext_dc,
        len(cell_types),
        lines_total,
        num_ct_extrapolated,
        region_str=region_str,
        ann_set_name=ann_set_name,
        output_filename=plot_fn,
        all_ext_dcs = all_ext_dcs
    )

    return len(cell_types), num_ct_annotated, num_ct_extrapolated, rmse_mean, rmse_std, rmse_ext_mean, rmse_ext_std
