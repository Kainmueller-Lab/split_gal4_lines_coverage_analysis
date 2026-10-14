import os
import argparse
import logging
import numpy as np
from neuprint import Client, fetch_neurons, fetch_roi_hierarchy

# from neuprint import NeuronCriteria as NC
import pandas as pd
import json
import csv

try:
    from utils import load_config, save_dict_to_pickle
    from plot_utils import plot_num_lines_per_ct, plot_num_ct_per_line
    from simulate import simulate
except ModuleNotFoundError:
    from src.utils import load_config, save_dict_to_pickle
    from src.plot_utils import plot_num_lines_per_ct, plot_num_ct_per_line
    from src.simulate import simulate


logger = logging.getLogger(__name__)

em_dataset = {"brain": "hemibrain:v1.2.1", "vnc": "manc:v1.2.1"}


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    # TODO use relative paths
    parser.add_argument("--output_folder", type=str, default=".")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
    )
    parser.add_argument("--cns_part", choices=["brain", "vnc"], required=True)
    parser.add_argument("--ann", choices=["true", "all", "both"], default="true")
    parser.add_argument("--region_key", type=str, default=None)
    parser.add_argument(
        "--lines_total", type=int, default=3060, help="total number of lines"
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--consolidate_types", action="store_true")
    parser.add_argument(
        "--subset_size",
        type=float,
        default=0.5,
        help="Fraction of lines to sample in each subset (simulate)",
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=5,
        help="Number of subsets to sample (simulate)",
    )
    parser.add_argument(
        "--num_sample_obs",
        type=int,
        default=100,
        help="Number of MC samples for observed discovery curve (simulate)",
    )
    parser.add_argument(
        "--num_sample_opt",
        type=int,
        default=100,
        help="Number of MC samples for optimization (simulate)",
    )
    parser.add_argument(
        "--num_sample_ext",
        type=int,
        default=100,
        help="Number of MC samples for extrapolation (simulate)",
    )
    parser.add_argument(
        "--num_iter_opt",
        type=int,
        default=20,
        help="Number of optimization iterations (simulate), must be >= 11",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for simulation (default: serial)",
    )

    args = parser.parse_args()
    return args


def consolidate_subtypes(cell_type_list):
    logger.info(f"consolidating {len(cell_type_list)} cell types")
    # read in cell type mapping json
    with open(cell_type_mapping_file, "r") as f:
        ct_mapping = json.load(f)

    subct_to_ct = ct_mapping["subct_to_ct"]
    # ct_to_subct = ct_mapping["ct_to_subct"]
    cell_type_list = [
        ct if ct not in subct_to_ct.keys() else subct_to_ct[ct] for ct in cell_type_list
    ]
    cell_type_list = list(np.unique(cell_type_list))
    logger.info(f"consolidated to {len(cell_type_list)} cell types")
    return cell_type_list


def update_annotations_for_consolidation(ann_df, ct_mapping_fn, grouped_ann_fn):
    # read in revised annotations
    rev_df = pd.read_excel(grouped_ann_fn)
    logger.info(f"ann_df: {len(ann_df)}, rev_df: {len(rev_df)}")

    # check if there are lines in rev_df that are not in ann_df
    # Step 1: Get unique "Line Name" values from both DataFrames
    line_names_a = set(ann_df["Line Name"].dropna().unique())
    line_names_b = set(rev_df["Line Name"].dropna().unique())

    # Step 2: Find Line Names present in df_b but not in df_a
    line_names_only_in_b = line_names_b - line_names_a

    # Step 3: Convert to sorted list (optional)
    line_names_only_in_b = sorted(list(line_names_only_in_b))

    assert len(line_names_only_in_b) == 0, (
        "Line Names in rev_df but not in ann_df: %s" % line_names_only_in_b
    )

    # check for consistent annotation
    # overwrite sub cell type maybe with cell type true if given
    # already overwrite sub cell type with consolidated cell type
    # Step 1: Prepare rev_df for merge — rename columns to match ann_df semantics
    rev_df_renamed = rev_df.rename(
        columns={
            "Consolidated Cell Type": "Updated Cell types",
            "New Annotation": "Updated Annotation",
        }
    )

    # Step 2: Merge ann_df with rev_df on "Line Name" and "Cell types"
    merged_df = pd.merge(
        ann_df,
        rev_df_renamed[
            ["Line Name", "Cell types", "Updated Cell types", "Updated Annotation"]
        ],
        on=["Line Name", "Cell types"],
        how="left",
    )

    # Step 3: Conditionally update columns in ann_df
    merged_df["Cell types"] = merged_df.apply(
        lambda row: row["Updated Cell types"]
        if pd.notna(row["Updated Cell types"])
        and row["Cell types"] != row["Updated Cell types"]
        else row["Cell types"],
        axis=1,
    )

    merged_df["Annotation"] = merged_df.apply(
        lambda row: row["Updated Annotation"]
        if pd.notna(row["Updated Annotation"])
        and row["Annotation"] != row["Updated Annotation"]
        else row["Annotation"],
        axis=1,
    )

    # Step 4: Drop helper columns
    ann_df_updated = merged_df.drop(
        columns=["Updated Cell types", "Updated Annotation"]
    )

    # check for further cell type consolidations
    with open(cell_type_mapping_file, "r") as f:
        ct_mapping = json.load(f)
    subct_to_ct = ct_mapping["subct_to_ct"]
    ann_df_updated["Cell types"] = ann_df_updated["Cell types"].apply(
        lambda x: subct_to_ct[x] if x in subct_to_ct else x
    )

    # Step 5: Remove duplicates based on key columns
    ann_df_final = ann_df_updated.drop_duplicates(
        subset=["Line Name", "Random Sample", "Region", "Cell types", "Annotation"]
    )

    # Optional: Reset index if needed
    ann_df_final = ann_df_final.reset_index(drop=True)

    return ann_df_final


def get_neuprint_cell_types(region, np_regions, consolidate=False):
    # criteria = NC(rois=np_regions) # status="Traced"
    # neuron_df, _ = fetch_neurons(criteria)
    neuron_df, _ = fetch_neurons()

    if np_regions is not None:
        target_set = set(np_regions)
        # Create filters
        input_mask = neuron_df["inputRois"].apply(
            lambda rois: bool(set(rois) & target_set)
        )
        output_mask = neuron_df["outputRois"].apply(
            lambda rois: bool(set(rois) & target_set)
        )
        # Combine with OR to get any match
        neuron_df = neuron_df[input_mask | output_mask]
    num_neurons_unfiltered = len(neuron_df)

    # filter em neurons by size and type
    if region == "brain":
        cdf = neuron_df.query("cropped == True and size <= 200000000 and type.isnull()")
    else:
        cdf = neuron_df.query("size <= 100000000 and type.isnull()")
    neuron_df = neuron_df[~neuron_df.bodyId.isin(cdf.bodyId)]
    cell_types = neuron_df["type"].dropna().unique().tolist()
    if region == "brain" and consolidate:
        cell_types = consolidate_subtypes(cell_types)
    logger.info(
        f"num cell types: {len(cell_types)}, num em neurons unfiltered: {num_neurons_unfiltered}"
    )
    return cell_types


def estimate_coverage_for_all_lines(
    c_ann_lines,
    c_ann_cell_types,
    c_cell_types,
    c_ct_per_line,
    num_lines_per_ct,
    lines_total,
    cns_part,
    region_key,
    ann_set_name,
    output_folder="",
    skip_existing=False,
    args=None,
):
    num_ct_total, num_ct_annotated, num_ct_extrapolated, rmse_mean, rmse_std, rmse_ext_mean, rmse_ext_std = simulate(
        c_ann_lines,
        c_ann_cell_types,
        c_cell_types,
        c_ct_per_line,
        num_lines_per_ct,
        lines_total,
        cns_part,
        region_key,
        ann_set_name,
        subset_size=args.subset_size,
        num_subsets=args.num_subsets,
        num_sample_obs=args.num_sample_obs,
        num_sample_opt=args.num_sample_opt,
        num_sample_ext=args.num_sample_ext,
        num_iter_opt=args.num_iter_opt,
        output_folder=output_folder,
        skip_existing=skip_existing,
        num_workers=args.num_workers,
    )

    return num_ct_total, num_ct_annotated, num_ct_extrapolated, rmse_mean, rmse_std, rmse_ext_mean, rmse_ext_std


def get_coverage(
    ann_lines,
    ann_cts,
    ann_cts_only_true,
    cell_types,
    ann_ct_per_line,
    ann_ct_per_line_only_true,
    num_lines_per_ct,
    num_lines_per_ct_only_true,
    cns_part,
    region_key,
    neuprint_rois,
    lines_total,
    output_folder="",
    skip_existing=False,
    args=None,
):
    # ------------------------------------------------------------
    # coverage for only true annotations
    num_ct_total, num_ct_annotated, num_ct_extrapolated, rmse_mean, rmse_std, rmse_ext_mean, rmse_ext_std = (
        estimate_coverage_for_all_lines(
            ann_lines,
            ann_cts_only_true,
            cell_types,
            ann_ct_per_line_only_true,
            num_lines_per_ct_only_true,
            lines_total,
            cns_part,
            region_key,
            "true",
            output_folder=output_folder,
            skip_existing=skip_existing,
            args=args,
        )
    )
    # calculate coverage values
    output_true = {
        "region": region_key,
        "neuprint ROI": neuprint_rois,
        "num cell types total": num_ct_total,
        "num cell types annotated": num_ct_annotated,
        "coverage from annotated": float(num_ct_annotated) / num_ct_total,
        "num cell types simulated": num_ct_extrapolated,
        "coverage from simulated": float(num_ct_extrapolated) / num_ct_total,
        "rmse mean": rmse_mean,
        "rmse_std": rmse_std,
        "rmse_ext_mean": rmse_ext_mean,
        "rmse_ext_std": rmse_ext_std
    }

    # coverage for true+maybe annotations
    num_ct_total, num_ct_annotated, num_ct_extrapolated, rmse_mean, rmse_std, rmse_ext_mean, rmse_ext_std = (
        estimate_coverage_for_all_lines(
            ann_lines,
            ann_cts,
            cell_types,
            ann_ct_per_line,
            num_lines_per_ct,
            lines_total,
            cns_part,
            region_key,
            "true_maybe",
            output_folder=output_folder,
            skip_existing=skip_existing,
            args=args,
        )
    )
    # calculate coverage values
    output_true_maybe = {
        "region": region_key,
        "neuprint ROI": neuprint_rois,
        "num cell types total": num_ct_total,
        "num cell types annotated": num_ct_annotated,
        "coverage from annotated": float(num_ct_annotated) / num_ct_total,
        "num cell types simulated": num_ct_extrapolated,
        "coverage from simulated": float(num_ct_extrapolated) / num_ct_total,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "rmse_ext_mean": rmse_ext_mean,
        "rmse_ext_std": rmse_ext_std
    }

    return output_true, output_true_maybe


def main():
    # get command line arguments and config
    args = get_arguments()
    config = load_config(args.config)

    # set logging level
    logging.basicConfig(level=config["general"]["log_level"])
    logger.info(f"Args: {args}")
    logger.info(f"Config: {config}")

    # load neuprint regions
    cns_part = args.cns_part
    if cns_part == "brain":
        region_infn = config["data"].get("brain_regions", None)
    else:
        region_infn = config["data"].get("vnc_regions", None)
    assert region_infn is not None, "Please check path to region json in config file!"
    with open(region_infn) as inf:
        regions_dict = json.load(inf)
        print(regions_dict)
    
    np.random.seed(42)
    logger.info(f"cns_part: {cns_part}, em_dataset: {em_dataset[cns_part]}")
    neuprint_token = config["general"].get("neuprint_token", None)
    c = Client(
        "neuprint.janelia.org",
        dataset=em_dataset[cns_part],
        token=neuprint_token
    )
    # print region hierarchy from neuprint
    logger.debug(fetch_roi_hierarchy(False, mark_primary=True, format="text"))

    # read all annotations
    ann_df = pd.read_excel(config["data"][cns_part], index_col=None)
    # check if sub cell types should be consolidated to cell types for brain
    if cns_part == "brain" and args.consolidate_types:
        ct_mapping_fn = config["data"]["cell_type_mapping"]
        grouped_ann_fn = config["data"]["grouped_annotations_only_brain"]
        ann_df = update_annotations_for_consolidation(ann_df, ct_mapping_fn,
                grouped_ann_fn)
    # filter only as true annotated hits
    ann_df_only_true = ann_df.query("`Cleaned Annotation` == 'True'")
    # get list of annotated lines
    ann_lines = ann_df["Line Name"].dropna().unique().tolist()
    logger.info(f"Number of annotated lines: {len(ann_lines)}")

    # get all annotated cell types (true + maybe, only true)
    ann_cell_types = ann_df["Cell types"].dropna().value_counts()
    ann_cell_types_only_true = ann_df_only_true["Cell types"].dropna().value_counts()

    # ------------------------------------------------------------
    # get unique cell types per line
    ann_ct_per_line = {
        line: ann_df.query("`Line Name` == @line")["Cell types"]
        .dropna()
        .unique()
        .tolist()
        for line in ann_lines
    }
    ann_ct_per_line_only_true = {
        line: ann_df_only_true.query("`Line Name` == @line")["Cell types"]
        .dropna()
        .unique()
        .tolist()
        for line in ann_lines
    }
    # ------------------------------------------------------------
    # plot number of cell types per line for true and maybe annotations
    if (
        args.region_key is None
        or args.region_key == "hemibrain"
        or args.region_key == "MANC"
    ):
        # save data to pickle
        save_dict_to_pickle(
                os.path.join(args.output_folder, "num_ct_per_line.pkl"),
                {
                    "ann_lines": ann_lines,
                    "ann_ct_per_line": ann_ct_per_line,
                    "ann_ct_per_line_only_true": ann_ct_per_line_only_true
                    }
                )
        plot_num_ct_per_line(
            ann_lines,
            ann_ct_per_line,
            ann_ct_per_line_only_true,
            cns_part,
            output_folder=args.output_folder,
        )

    # ------------------------------------------------------------
    # get neuprint cell types
    cell_types = get_neuprint_cell_types(
        cns_part, None, consolidate=args.consolidate_types
    )

    # ------------------------------------------------------------
    # get number of hits per cell type
    num_lines_per_ct = {
        ct: sum([int(ct in ann_ct_per_line[line]) for line in ann_lines])
        for ct in cell_types
    }
    num_lines_per_ct_only_true = {
        ct: sum([int(ct in ann_ct_per_line_only_true[line]) for line in ann_lines])
        for ct in cell_types
    }

    # plot number of hits per cell type
    if (
        args.region_key is None
        or args.region_key == "hemibrain"
        or args.region_key == "MANC"
    ):
        # save data to pickle
        save_dict_to_pickle(
                os.path.join(args.output_folder, "ct_frequencies.pkl"),
                {
                    "num_lines_per_ct": num_lines_per_ct,
                    "num_lines_per_ct_only_true": num_lines_per_ct_only_true,
                    }
                )
        plot_num_lines_per_ct(
            num_lines_per_ct,
            num_lines_per_ct_only_true,
            cns_part,
            output_folder=args.output_folder,
        )

    # create placeholder for output
    df_ct_regions_all = None
    df_ct_regions_ann = None
    df_ct_regions_ann_only_true = None
    df_coverage = pd.DataFrame(
        columns=[
            "region",
            "neuprint ROI",
            "num cell types total",
            "num cell types annotated",
            "coverage from annotated",
            "num cell types estimated",
            "coverage from estimated",
        ]
    )
    df_coverage_only_true = pd.DataFrame(
        columns=[
            "region",
            "neuprint ROI",
            "num cell types total",
            "num cell types annotated",
            "coverage from annotated",
            "num cell types estimated",
            "coverage from estimated",
        ]
    )

    if args.region_key is not None:
        assert args.region_key in regions_dict, f"Given regions key is not valid: {args.region_key}, valid keys: {list(regions_dict.keys())}"
        regions_dict = {args.region_key: regions_dict[args.region_key]}
        logger.info(
            f"only processing {args.region_key}, regions_dict: {regions_dict}"
        )

    # ------------------------------------------------------------
    # iterate through regions and get coverage
    for region_key, neuprint_rois in regions_dict.items():
        # ------------------------------------------------------------
        # get neuprint cell types
        if neuprint_rois is not None:
            cell_types = get_neuprint_cell_types(
                cns_part, neuprint_rois, consolidate=args.consolidate_types
            )
        logger.info(f"Number of cell types in {neuprint_rois}: {len(cell_types)}")
        if len(cell_types) == 0:
            continue
        if len(cell_types) < 10:
            continue

        # filter annotated cell types depending on region
        if neuprint_rois is not None:
            ann_ct_in_region = ann_cell_types.filter(items=cell_types)
            ann_ct_in_region_only_true = ann_cell_types_only_true.filter(
                items=cell_types
            )
        else:
            ann_ct_in_region = ann_cell_types
            ann_ct_in_region_only_true = ann_cell_types_only_true
        ann_ct_in_region = list(ann_ct_in_region.keys())
        ann_ct_in_region_only_true = list(ann_ct_in_region_only_true.keys())

        # call coverage function
        cov_true, cov_true_maybe = get_coverage(
            ann_lines,
            ann_ct_in_region,
            ann_ct_in_region_only_true,
            cell_types,
            ann_ct_per_line,
            ann_ct_per_line_only_true,
            num_lines_per_ct,
            num_lines_per_ct_only_true,
            cns_part,
            region_key,
            neuprint_rois,
            args.lines_total,
            output_folder=args.output_folder,
            skip_existing=args.skip_existing,
            args=args,
        )

        # save results to global output variables
        df_coverage = pd.concat(
            [df_coverage, pd.DataFrame([cov_true_maybe])], ignore_index=True
        )
        df_coverage_only_true = pd.concat(
            [df_coverage_only_true, pd.DataFrame([cov_true])], ignore_index=True
        )

        # save total cell types per region
        if df_ct_regions_all is None:
            df_ct_regions_all = pd.DataFrame({region_key: cell_types})
        else:
            df_ct_regions_all = pd.concat(
                [df_ct_regions_all, pd.DataFrame({region_key: cell_types})], axis=1
            )
        # save true+maybe cell types per region
        if df_ct_regions_ann is None:
            df_ct_regions_ann = pd.DataFrame({region_key: ann_ct_in_region})
        else:
            df_ct_regions_ann = pd.concat(
                [df_ct_regions_ann, pd.DataFrame({region_key: ann_ct_in_region})],
                axis=1,
            )
        # save true cell types per region
        if df_ct_regions_ann_only_true is None:
            df_ct_regions_ann_only_true = pd.DataFrame(
                {region_key: ann_ct_in_region_only_true}
            )
        else:
            df_ct_regions_ann_only_true = pd.concat(
                [
                    df_ct_regions_ann_only_true,
                    pd.DataFrame({region_key: ann_ct_in_region_only_true}),
                ],
                axis=1,
            )

    # save to file
    outfn = os.path.join(args.output_folder, "coverages_%s.xlsx" % cns_part)
    with pd.ExcelWriter(outfn) as xlsx_writer:
        df_coverage_only_true.to_excel(
            xlsx_writer, sheet_name="coverage only true", index=False
        )
        df_coverage.to_excel(xlsx_writer, sheet_name="coverage true+maybe", index=False)
        df_ct_regions_ann_only_true.to_excel(
            xlsx_writer, sheet_name="true annotated types per region", index=False
        )
        df_ct_regions_ann.to_excel(
            xlsx_writer, sheet_name="true+maybe annotated types per region", index=False
        )
        df_ct_regions_all.to_excel(
            xlsx_writer, sheet_name="total types per region", index=False
        )


if __name__ == "__main__":
    main()
