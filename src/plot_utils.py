import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_lines_from_dict(
    xdata_dict,
    ydata_dict,
    color_dict=None,
    linestyle_dict=None,
    num_ct_annotated=None,
    num_lines_annotated=None,
    output_filename="plot_lines_from_dict.png",
    subset_size=None,
    trainset_size=None,
    testset_error_mean=None,
    testset_error_std=None
):
    plt.clf()
    plt.figure(figsize=(8, 5))
    ymax = 0
    for k in xdata_dict.keys():
        if k == "obs":
            hobs = plt.plot(
                    xdata_dict[k],
                    ydata_dict[k],
                    label="Mean (observed)",
                    color=color_dict.get(k, None) if color_dict else None,
                    linestyle=linestyle_dict.get(k, None) if linestyle_dict else None,
                    )[0]
        elif k == "model_example":
            hexp = plt.plot(
                    xdata_dict[k],
                    ydata_dict[k],
                    label="Mean per model (simulated)",
                    color=color_dict.get(k, None) if color_dict else None,
                    linestyle=linestyle_dict.get(k, None) if linestyle_dict else None,
                    )[0]
        else:
            plt.plot(
                xdata_dict[k],
                ydata_dict[k],
                label=k,
                color=color_dict.get(k, None) if color_dict else None,
                linestyle=linestyle_dict.get(k, None) if linestyle_dict else None,
            )
        if np.max(ydata_dict[k]) > ymax:
            ymax = np.max(ydata_dict[k])
    if num_ct_annotated is not None and num_lines_annotated is not None:
        plt.plot(
            num_lines_annotated,
            num_ct_annotated,
            marker="*",
            markersize=12,
            color="black",
        )
    if subset_size is not None:
        plt.vlines(subset_size, ymin=0, ymax=ymax, color='gray', linestyle="--")
    if trainset_size is not None:
        plt.vlines(trainset_size, ymin=0, ymax=ymax, color='gray', linestyle="-")
    if trainset_size is not None and num_lines_annotated is not None:
        plt.vlines(num_lines_annotated, ymin=0, ymax=ymax, color='gray', linestyle="-")
    if testset_error_mean is not None and testset_error_std is not None:
        plt.title("Mean simulated discovery curves from all models with test error of %.2f +/- %.2f" % (testset_error_mean, testset_error_std))
    handles = []
    if "obs" in list(xdata_dict.keys()):
        handles = [hobs]
    if "model_example" in list(xdata_dict.keys()):
        handles += [hexp]
    print("handles: ", handles)
    if len(handles) > 0:
        plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_observed_discovery_curve(
    x, mean, lower, upper, max_cell_types, region_str, ann_set_name, output_filename
):
    """
    Plot the observed discovery curve as the percentage of unique cell types found per number of lines.

    Parameters:
        x (array-like): Number of lines (x-axis values).
        mean (array-like): Mean discovery curve (number of unique cell types per line).
        lower (array-like): Lower bound of the 95% confidence interval.
        upper (array-like): Upper bound of the 95% confidence interval.
        max_cell_types (int): Total number of unique cell types (for normalization).
        region_str (str): Region identifier (used in plot title).
        ann_set_name (str): Annotation set identifier ('true' or 'true_maybe').
        output_filename (str): Path to save the plot image.

    Saves:
        PNG plot of the normalized observed discovery curve with confidence interval.
    """
    if ann_set_name == "true":
        ann_str = "confident hits"
    elif ann_set_name == "true_maybe":
        ann_str = "all hits including candidates"
    else:
        raise NotImplementedError
    # get percentage of unique cell types
    mean = np.array(mean) / max_cell_types
    lower = np.array(lower) / max_cell_types
    upper = np.array(upper) / max_cell_types

    # create the plot
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label="Mean discovery curve")
    plt.fill_between(x, lower, upper, alpha=0.2, label="95% Confidence interval")
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.xlabel("Number of lines")
    plt.ylabel("Percentage of unique cell types (%i)" % max_cell_types)
    # plt.ylabel("Number of unique cell types")
    plt.title("Observed discovery curves for %s (%s)" % (ann_str, region_str))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_observed_and_extrapolated_discovery_curves(
    obs_x,
    obs_mean,
    obs_lower,
    obs_upper,
    exp_x,
    exp_mean,
    exp_lower,
    exp_upper,
    max_ct,
    max_lines,
    num_ct_max_lines,
    region_str,
    ann_set_name,
    output_filename,
    all_ext_dcs = None,
    **kwargs,
):
    """
    Plot observed and extrapolated (simulated) discovery curves as the percentage of unique cell types found per number of lines.

    Parameters:
        obs_x (array-like): X-axis values for observed discovery curve (number of lines).
        obs_mean (array-like): Mean observed discovery curve (unique cell types per line).
        obs_lower (array-like): Lower bound of 95% confidence interval for observed curve.
        obs_upper (array-like): Upper bound of 95% confidence interval for observed curve.
        exp_x (array-like): X-axis values for extrapolated discovery curve.
        exp_mean (array-like): Mean extrapolated discovery curve.
        exp_lower (array-like): Lower bound of 95% confidence interval for extrapolated curve.
        exp_upper (array-like): Upper bound of 95% confidence interval for extrapolated curve.
        max_ct (int): Total number of unique cell types (for normalization).
        max_lines (int): Total number of lines (for reference/support lines).
        num_ct_max_lines (int): Number of cell types at max lines (for reference/support lines).
        region_str (str): Region identifier (used in plot title).
        ann_set_name (str): Annotation set identifier ('true' or 'true_maybe').
        output_filename (str): Path to save the plot image.
        **kwargs: Additional keyword arguments for advanced plotting (not used in default version).

    Saves:
        PNG plot of observed and extrapolated discovery curves with confidence intervals.
    """
    if ann_set_name == "true":
        ann_str = "confident hits"
    elif ann_set_name == "true_maybe":
        ann_str = "all hits including candidates"
    else:
        raise NotImplementedError

    if all_ext_dcs is not None:
        p2_5  = np.array(np.percentile(all_ext_dcs, 2.5,  axis=0), dtype=float) / max_ct
        p25   = np.array(np.percentile(all_ext_dcs, 25,   axis=0), dtype=float) / max_ct
        p50   = np.array(np.percentile(all_ext_dcs, 50,   axis=0), dtype=float) / max_ct
        p75   = np.array(np.percentile(all_ext_dcs, 75,   axis=0), dtype=float) / max_ct
        p97_5 = np.array(np.percentile(all_ext_dcs, 97.5, axis=0), dtype=float) / max_ct

    # get percentage of unique cell types
    exp_mean = np.array(exp_mean, dtype=float) / max_ct
    exp_lower = np.array(exp_lower, dtype=float) / max_ct
    exp_upper = np.array(exp_upper, dtype=float) / max_ct
    obs_mean = np.array(obs_mean, dtype=float) / max_ct
    obs_lower = np.array(obs_lower, dtype=float) / max_ct
    obs_upper = np.array(obs_upper, dtype=float) / max_ct

    # plot observed and expected discovery curves
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    # Uncertainty ribbons (draw widest first)
    h95 = ax.fill_between(exp_x, p2_5,  p97_5, alpha=0.2, color="orange", label="95% band")
    #ax.fill_between(x, p10,   p90,   alpha=0.25, label="80% band")
    h50 = ax.fill_between(exp_x, p25,   p75,   alpha=0.25, color="orange", label="50% band")

    #plt.figure(figsize=(8, 5))
    h1 = ax.plot(
        exp_x, exp_mean, color="orange", label="Mean (simulated)"
    )[0]


    #h2 = plt.fill_between(
    #    exp_x,
    #    exp_lower,
    #    exp_upper,
    #    alpha=0.2,
    #    color="orange",
    #    label="95% Confidence interval",
    #)
    h3 = ax.plot(obs_x, obs_mean, color="darkblue", linestyle="--", label="Mean (observed)")[
        0
    ]

    h_ex = ax.axvspan(len(obs_x), max_lines, alpha=0.04, label="Extrapolated region")

    # h4 = plt.fill_between(
    #             obs_x,
    #             obs_lower,
    #             obs_upper,
    #             alpha=0.2,
    #             color="darkblue",
    #             label="95% Confidence interval"
    #             )

    # add support lines
    # ymax_total_lines = num_ct_max_lines if ann_set_str != "both" else \
    #        kwargs["num_ct_max_lines_all"]
    # h1 = plt.vlines(max_lines, ymin=0, ymax=ymax_total_lines, color='gray',
    #        label="Total lines (%i)" % max_lines)
    # h2 = plt.hlines(y=max_ct, xmin=0, xmax=exp_x[-1],
    #        colors='green', linestyles='-',
    #        label="Max cell types (%i)" % max_ct)
    # plt.hlines(y=num_ct_max_lines, xmin=0, xmax=max_lines, colors='gray',
    #        linestyles="dashed")

    # if ann_set_str == "both":
    #    exp_c2 = plt.plot(exp_x, kwargs["exp_dc_all"], color="cornflowerblue",
    #            label="Estimate for all potential hits")[0]
    #    obs_c2 = plt.plot(obs_x, kwargs["obs_mean_all"], color="orange",
    #            label="Observed for all potential hits")[0]
    #    #plt.hlines(y=kwargs["num_ct_max_lines_all"], xmin=0, xmax=max_lines, colors='gray',
    #    #        linestyles="dashed")
    #    handles=[obs_c1, obs_c2, h2, exp_c1, exp_c2, h1]
    # else:
    handles = [h3, h1, h50, h95, h_ex]
    ax.set_ylim(0, 0.65)

    ax.set_xlabel("Number of lines")
    ax.set_ylabel("Percentage of unique cell types (%i)" % max_ct)
    ax.set_title("Extrapolated discovery curves for %s (%s)" % (ann_str, region_str))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    # ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
    #          loc='lower left', fontsize='small')
    # plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
    #        ncol=2, frameon=False)
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_simulated_all_alphas(dcs, obs_dc, output_filename):
    plt.clf()
    plt.figure(figsize=(8, 5))
    for k, dc in dcs.items():
        print(k, dc.keys())
        plt.plot(dc["x"], dc["mean"], label="Simulated with alpha %s" % str(k))
        label_str = "95% confident interval for sim. with alpha " + str(k)
        print(label_str, type(label_str))
        plt.fill_between(dc["x"], dc["lower"], dc["upper"], alpha=0.2, label=label_str)
    # plot observed discovery curve
    plt.plot(obs_dc["x"], obs_dc["mean"], label="Observed discovery curve", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_expected_and_simulated(
    sim_x,
    sim_mean,
    sim_lower,
    sim_upper,
    exp_x,
    exp_mean,
    max_ct,
    max_lines,
    num_ct_max_lines,
    region_str,
    ann_set_str,
    output_filename,
):
    plt.clf()
    plt.figure(figsize=(8, 5))
    # plot expected and simulated discovery curves
    plt.plot(
        sim_x,
        sim_mean,
        color="blue",
        linewidth=2,
        label="Simulated for high-confident hits",
    )
    plt.fill_between(
        sim_x,
        sim_lower,
        sim_upper,
        label="95% confident interval for simulated",
        alpha=0.2,
    )
    plt.plot(exp_x, exp_mean, color="orange", label="Estimate for high-confident hits")

    # add support lines
    # plt.vlines(max_lines, ymin=0, ymax=num_ct_max_lines, color='gray',
    #        label="Total lines (%i)" % max_lines)
    # plt.hlines(y=max_ct, xmin=0, xmax=exp_x[-1],
    #        colors='green', linestyles='-', label="Max cell types (%i)" % max_ct)
    # plt.hlines(y=num_ct_max_lines, xmin=0, xmax=max_lines, colors='gray',
    #        linestyles="dashed")

    plt.xlabel("Number of lines")
    plt.ylabel("Cumulative unique cell types")
    plt.title("Simulated discovery curve in %s" % region_str)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_filename)
    plt.close()


def plot_cov_per_region(df, cns_part, output_filename, title=None, add_both=False):
    # -------- Sort and Normalize --------
    if cns_part == "vnc":
        cns_key = "MANC"
    elif cns_part == "brain":
        cns_key = "hemibrain"
    df_complete = df[df["region"] == cns_key]
    df_sorted_rest = df[df["region"] != cns_key].sort_values("num cell types total",
            ascending=True)
    df_sorted = pd.concat([df_sorted_rest, df_complete], ignore_index=True)
    
    # -------- Labels: "Region Name: N" --------
    #labels = df_sorted.apply(
    #    lambda r: f"{r['region name']} ({r['region']}): {int(r['num cell types total'])}",
    #    axis=1
    #)
    labels = df_sorted.apply(
        lambda r: (
            rf"$\bf{{{r['region name']}}}$: {int(r['num cell types total'])}"
            if r["region"] == "hemibrain" or r["region"] == "MANC"
            else f"{r['region name']} ({r['region']}): {int(r['num cell types total'])}"
        ),
        axis=1,
    )

    if title is None:
        title = "Extrapolated Coverage by Region for %s" % cns_key.upper()
    
    #norm = Normalize(
    #    vmin=df_sorted["num cell types total"].min(),
    #    vmax=df_sorted["num cell types total"].max(),
    #)
    #cmap = plt.cm.viridis
    #colors = cmap(norm(df_sorted["num cell types total"].values))

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(12, 6))
    if add_both:
        true_maybes = ax.barh(
            labels,
            df_sorted["coverage from simulated all"],
            #color="orange",
            alpha=0.6,
            label="All hits incl. candidates",
        )
    trues = ax.barh(
        labels,
        df_sorted["coverage from simulated"],
        color="orange",
        #alpha=0.6,
        label="High-confident hits",
    )

    # -------- Annotated coverage markers (red lines) --------
    #y_positions = range(len(df_sorted))

    #annotated_lines = ax.vlines(
    #    df_sorted["coverage from annotated"],
    #    [y - 0.4 for y in y_positions],
    #    [y + 0.4 for y in y_positions],
    #    color="red",
    #    linewidth=2,
    #    label="Annotated coverage",
    #)

    # Red vertical lines for annotated coverage
    # for i, (coverage, region) in enumerate(
    #        zip(df_sorted['coverage from annotated'], df_sorted['region'])):
    #    ax.plot([coverage, coverage], [i - 0.4, i + 0.4],
    #            color='red', linewidth=1, label='Annotated' if i == 0 else "")

    # -------- Text labels at the end of bars --------
    # for y_pos, (idx, row) in enumerate(df_sorted.iterrows()):
    #    ax.text(1.01, y_pos, f"{int(row['num cell types total'])}", va='center', fontsize=8)
    #for i, (ct_total, bar_color) in enumerate(
    #    zip(df_sorted["num cell types total"], colors)
    #):
    #    # Determine luminance to choose black or white text
    #    r, g, b, _ = bar_color
    #    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    #    text_color = "black" if luminance > 0.6 else "white"
    #    ax.text(
    #        0.01,
    #        i,
    #        f"{int(ct_total)}",
    #        va="center",
    #        ha="left",
    #        color=text_color,
    #        fontsize=8,
    #    )

    # -------- Colorbar --------
    #sm = ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #fig.colorbar(sm, ax=ax, label="Total Cell Types")

    # -------- Labels and formatting --------
    ax.set_xlabel("Coverage (%)")
    # ax.set_xlim(0, 1.2)
    ax.set_title(title)
    if add_both:
        ax.legend(handles=[trues, true_maybes], loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.40)

    # fig.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def plot_num_lines_per_ct(
    num_lines_per_ct, num_lines_per_ct_only_true, cns_part, output_folder="."
):
    """
    Plot the distribution of the number of occurrences (hits) per cell type in annotated lines for a given CNS part.

    Parameters:
        num_lines_per_ct (dict): Dictionary mapping cell type to number of occurrences (all hits including candidates).
        num_lines_per_ct_only_true (dict): Dictionary mapping cell type to number of occurrences (confident hits only).
        cns_part (str): CNS part, either 'vnc' or 'brain'.
        output_folder (str, optional): Directory to save the plot. Defaults to current directory.

    Saves:
        Bar plot as 'plot_num_lines_per_ct_<cns_part>.png' in the output folder.
    """
    if cns_part == "vnc":
        region_str = "MANC"
    else:
        region_str = "Hemibrain"
    plt.clf()
    num_cts = np.array(list(num_lines_per_ct.values()))
    num_cts_only_true = np.array(list(num_lines_per_ct_only_true.values()))

    ydata = []
    ydata_only_true = []
    xlabels = ["1", "2", "3", "4", "5-10", ">10"]
    num_bins = len(xlabels)
    for i in range(1, 5):
        ydata.append(np.sum(num_cts == i))
        ydata_only_true.append(np.sum(num_cts_only_true == i))

    ydata.append(np.sum(np.logical_and(num_cts >= 5, num_cts <= 10)))
    ydata.append(np.sum(num_cts > 10))
    ydata_only_true.append(
        np.sum(np.logical_and(num_cts_only_true >= 5, num_cts_only_true <= 10))
    )
    ydata_only_true.append(np.sum(num_cts_only_true > 10))

    print(ydata, ydata_only_true)

    # plot number of hits per cell type
    b1 = plt.bar(
        np.arange(1 + 0.2, num_bins + 1 + 0.2, 1),
        ydata,
        label="All hits incl. candidates",
        width=0.4,
        alpha=0.6,
    )
    b2 = plt.bar(
        np.arange(1 - 0.2, num_bins + 1 - 0.2, 1),
        ydata_only_true,
        label="Confident hits",
        width=0.4,
        alpha=0.6,
    )
    plt.xlabel("Frequency of cell types")
    plt.xticks(ticks=range(1, num_bins + 1), labels=xlabels)
    plt.ylabel("Number of cell types")
    plt.legend(handles=[b2, b1])
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.title(
        "Frequency of annotated cell types across lines (%s)" % region_str
    )
    plt.tight_layout()
    plt.savefig(output_folder + "/plot_num_lines_per_ct_%s.png" % cns_part)
    plt.close()


def plot_num_ct_per_line(
    ann_lines, ann_ct_per_line, ann_ct_per_line_only_true, cns_part, output_folder="."
):
    """
    Plot the distribution of the number of annotated cell types per line for a given CNS part.

    Parameters:
        ann_lines (list): List of annotated line names.
        ann_ct_per_line (dict): Mapping from line name to list of cell types (true+maybe).
        ann_ct_per_line_only_true (dict): Mapping from line name to list of confident cell types.
        cns_part (str): CNS part, either 'vnc' or 'brain'.
        output_folder (str): Directory to save the plot.

    Saves:
        Bar plot as 'plot_num_ct_per_line_<cns_part>.png' in the output folder.
    """
    if cns_part == "vnc":
        region_str = "MANC"
    else:
        region_str = "Hemibrain"

    # get number of unique cell types per line
    num_ann_ct_per_line = {line: len(ann_ct_per_line[line]) for line in ann_lines}
    num_ann_ct_per_line_only_true = {
        line: len(ann_ct_per_line_only_true[line]) for line in ann_lines
    }

    # plot number of cell types per line for all annotations
    plt.clf()
    # number of cell types per line
    num_cts = np.unique(list(num_ann_ct_per_line.values()))
    # num_cts_only_true, cnts_only_true = np.unique(
    #        list(num_ann_ct_per_line_only_true.values()), return_counts=True)
    all_ct = []
    only_true = []
    num_ct_per_line = np.array(list(num_ann_ct_per_line.values()))
    num_ct_per_line_ot = np.array(list(num_ann_ct_per_line_only_true.values()))
    max_cts = max(num_cts)
    for i in range(max_cts):
        all_ct.append(np.sum(num_ct_per_line == i))
        only_true.append(np.sum(num_ct_per_line_ot == i))
    # all_ct = np.array(all_ct, dtype=float) / len(ann_lines)
    # only_true = np.array(only_true, dtype=float) / len(ann_lines)
    all_ct = np.array(all_ct)
    only_true = np.array(only_true)

    ba = plt.bar(
        np.arange(0.2, 0.2 + max_cts, 1),
        all_ct,
        label="All hits incl. candidates",
        alpha=0.6,
        width=0.4,
    )
    bt = plt.bar(
        np.arange(-0.2, -0.2 + max_cts, 1),
        only_true,
        label="Confident hits",
        alpha=0.6,
        width=0.4,
    )

    mean_all_ct = np.round(np.mean(list(num_ann_ct_per_line.values())), 2)
    std_all_ct = np.round(np.std(list(num_ann_ct_per_line.values())), 2)
    mean_only_true = np.round(np.mean(list(num_ann_ct_per_line_only_true.values())), 2)
    std_only_true = np.round(np.std(list(num_ann_ct_per_line_only_true.values())), 2)

    ymax = max(only_true)
    linewidth = 2
    ma1 = plt.vlines(
        mean_all_ct,
        ymin=0,
        ymax=ymax,
        color="darkblue",
        linewidth=linewidth,
        linestyles="dashed",
        label="All hits incl. candidates, mean: %.2f +/- %.2f"
        % (mean_all_ct, std_all_ct),
    )
    mt1 = plt.vlines(
        mean_only_true,
        ymin=0,
        ymax=ymax,
        color="darkorange",
        linewidth=linewidth,
        linestyles="dashed",
        label="Confident hits, mean: %.2f +/- %.2f" % (mean_only_true, std_only_true),
    )
    plt.xticks(ticks=range(max_cts))

    plt.xlabel("Number of annotated cell types per line")
    # plt.ylabel("Percentage of %i annotated lines" % len(ann_lines))
    plt.ylabel("Number of lines")
    plt.title("Number of annotated cell types per line (%s)" % region_str)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(handles=[bt, mt1, ba, ma1])
    plt.tight_layout()
    plt.savefig(output_folder + "/plot_num_ct_per_line_%s.png" % cns_part)
    plt.close()

