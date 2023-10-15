__author__ = "Abdurrahman Abul-Basher"
__date__ = "10/17/2023"
__copyright__ = "Copyright 2023, The Lee Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman Abul-Basher"
__email__ = "ar.basher@childrens.harvard.edu"
__status__ = "Production"
__description__ = "This file is the main entry to perform subtype detection."

import datetime
import json
import os
import sys
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    if sys.platform.startswith("win"):
        os.system("cls")
    else:
        os.system("clear")
    print("# " + "=" * 50)
    print("Author: " + __author__)
    print("Copyright: " + __copyright__)
    print("License: " + __license__)
    print("Version: " + __version__)
    print("Maintainer: " + __maintainer__)
    print("Email: " + __email__)
    print("Status: " + __status__)
    print("Date: " + datetime.datetime.strptime(__date__, "%m/%d/%Y").strftime("%B-%d-%Y"))
    print("Description: " + textwrap.TextWrapper(width=45, subsequent_indent="\t     ").fill(__description__))
    print("# " + "=" * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, "config.json"), "wt") as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    args = Arguments()

    ###***************************          Path arguments          ***************************###
    args.dspath = parse_args.dspath
    args.rspath = parse_args.rspath

    ###***************************          File arguments          ***************************###
    args.file_name = parse_args.file_name
    args.suptitle_name = parse_args.suptitle_name

    ###***************************         Global arguments         ***************************###
    args.methods = parse_args.methods
    args.num_jobs = parse_args.num_jobs
    args.export_spring = parse_args.export_spring

    ###***************************       Inference arguments        ***************************###
    args.exponentiate = parse_args.exponentiate  # See list of data
    args.direction = parse_args.direction
    args.iqr_range = parse_args.iqr_range
    args.normalize = parse_args.normalize
    args.q = parse_args.q
    # OutlierSumStatistic
    args.two_sided_test = parse_args.two_sided_test
    # DIDS
    args.dids_scoref = parse_args.dids_scoref
    # Seurat
    args.seurat_log_transform = parse_args.seurat_log_transform  # See list of data
    # PHet
    args.num_subsamples = parse_args.num_subsamples
    args.disp_type = parse_args.disp_type
    args.calculate_deltadisp = parse_args.calculate_deltadisp
    args.calculate_deltamean = parse_args.calculate_deltamean
    args.calculate_fisher = parse_args.calculate_fisher
    args.calculate_disc_power = parse_args.calculate_disc_power
    args.bin_pvalues = parse_args.bin_pvalues
    args.feature_weight = parse_args.feature_weight
    args.weight_range = parse_args.weight_range

    ###***************************        Evaluate arguments        ***************************###
    args.alpha = parse_args.alpha
    args.sort_by_pvalue = parse_args.sort_by_pvalue
    args.score_metric = parse_args.score_metric

    ###***************************    UMAP & Clustering arguments   ***************************###
    args.standardize = parse_args.standardize
    args.top_k_features = parse_args.top_k_features
    args.plot_top_k_features = parse_args.plot_top_k_features
    if not args.sort_by_pvalue:
        args.plot_top_k_features = True
    args.min_dist = parse_args.min_dist
    args.num_neighbors = parse_args.num_neighbors
    args.perform_cluster = parse_args.perform_cluster
    args.cluster_type = parse_args.cluster_type
    args.max_clusters = parse_args.max_clusters
    args.heatmap_plot = parse_args.heatmap_plot

    return args


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run subtypes detection algorithms.")

    args = Arguments()
    # Arguments for file paths
    parser.add_argument("--dspath", default=fph.DATASET_PATH, type=str,
                        help="Path to the dataset after the samples are processed. "
                             "The default is set to dataset folder outside the source code.")
    parser.add_argument("--rspath", default=fph.RESULT_PATH, type=str,
                        help="Path to the results. The default is set to result "
                             "folder outside the source code.")

    # Arguments for file names and models
    parser.add_argument("--file-name", type=str, default="srbct",
                        help="The file name to save an object.")
    # parser.add_argument("--file-name", type=str, required=True,
    #                     help="The file name to save an object.")
    parser.add_argument("--suptitle-name", type=str, default="OMICS",
                        help="The name of the suptitle of the figures. (default: 'OMICS')")

    # Global arguments
    parser.add_argument("--methods", nargs="+", type=list,
                        default=["ttest_g", "wilcoxon_g", "ks_g", "limma_g", "copa", "os", "ort", "most", "lsoss",
                                 "dids", "deco", "phet_br"],
                        choices=["ttest_p", "ttest_g", "wilcoxon_p", "wilcoxon_g", "ks_p", "ks_g", "limma_p",
                                 "limma_g", "dispersion_a", "dispersion_c", "deltadispersion", "deltadispersionmean",
                                 "iqr_a", "iqr_c", "deltaiqr", "deltaiqrmean", "copa", "os", "ort", "most", "lsoss",
                                 "dids", "deco", "phet_bd", "phet_br"],
                        help="Select subtype methods. (default: ['phet_br']).")
    parser.add_argument("--num-jobs", type=int, default=2,
                        help="Number of parallel workers. (default: 2).")
    parser.add_argument("--export-spring", action="store_true", default=False,
                        help="Whether or not to export data for the SPRING plot. (default: False).")

    # Arguments for inference
    parser.add_argument("--exponentiate", action="store_true", default=False,
                        help="Whether to exponentiate data or not. (default: False).")
    parser.add_argument("--direction", type=str, default="both",
                        help="Direction to compute the nonparametric permutation test. "
                             "Possible values are: up, down, and both. (default: up).")
    parser.add_argument("--iqr-range", type=tuple, default=(25, 75),
                        help="Two-element sequence containing floats in range of [0,100]. "
                             "Percentiles over which to compute the range. Each must be "
                             "between 0 and 100, inclusive. (default: (25, 75)).")
    parser.add_argument("--normalize", type=str, default="zscore",
                        help="Type of normalization. Possible values are: zscore and robust. "
                             "(default: zscore).")
    parser.add_argument("--q", type=float, default=75.0,
                        help="Percentile to compute, which must be between 0 and 100 inclusive. "
                             "(default: 75.0).")
    # OutlierSumStatistic
    parser.add_argument("--two-sided-test", action="store_true", default=False,
                        help="Whether or not to compute the two sided test for OutlierSumStatistic. "
                             "(default: False).")
    # DIDS
    parser.add_argument("--dids-scoref", type=str, default="tanh", choices=["tanh", "sqrt", "quad"],
                        help="DIDS scoring function. Possible values are: tanh, sqrt, and quad. "
                             "(default: tanh).")
    # Seurat
    parser.add_argument("--seurat-log-transform", action="store_true", default=False,
                        help="Apply log transformation for dispersion based analysis. (default: False).")
    # PHet
    parser.add_argument("--num-subsamples", type=int, default=1000,
                        help="The number of subsamples for PHet. (default: 1000).")
    parser.add_argument("--disp-type", type=str, default="iqr", choices=["hvf", "iqr"],
                        help="Type of the dispersion based features calculation for PHet. "
                             "(default: 'iqr').")
    parser.add_argument("--calculate-deltadisp", action="store_false", default=True,
                        help="Whether or not to compute delta dispersion for PHet. "
                             "(default: True).")
    parser.add_argument("--calculate-deltamean", action="store_true", default=False,
                        help="Whether or not to compute delta mean for PHet. "
                             "(default: False).")
    parser.add_argument("--calculate-fisher", action="store_false", default=True,
                        help="Whether or not to compute Fisher\"s method for PHet. "
                             "(default: True).")
    parser.add_argument("--calculate-disc-power", action="store_false", default=True,
                        help="Whether or not to compute discriminative power for features. "
                             "(default: True).")
    parser.add_argument("--bin-pvalues", action="store_false", default=True,
                        help="Whether to apply binning pvalues from the KS test for PHet. "
                             "(default: True).")
    parser.add_argument("--feature-weight", nargs="+", type=list,
                        default=[0.4, 0.3, 0.2, 0.1],
                        help="Four hyper-parameters for PHet. "
                             "(default: [0.4, 0.3, 0.2, 0.1]).")
    parser.add_argument("--weight-range", nargs="+", type=list,
                        default=[0.1, 0.4, 0.8],
                        help="Three hyper-parameters to give a weight for a feature. This is only used "
                             "when the '--bin-pvalues' is set to false. "
                             "(default: [0.1, 0.4, 0.8]).")

    # Arguments for evaluation
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Significance level. (default: 0.01).")
    parser.add_argument("--sort-by-pvalue", action="store_false", default=True,
                        help="Whether rank features scores or sort pvalues using Gamma distribution. "
                             "(default: True).")
    parser.add_argument("--score-metric", type=str, default="f1",
                        choices=["f1", "precision", "recall", "auc", "accuracy", "jaccard"],
                        help="The score metric for evaluation. (default: f1).")

    # Arguments for clustering and UMAP
    parser.add_argument("--standardize", action="store_false", default=True,
                        help="Whether to standardize the data for UMAP or not. (default: True).")
    parser.add_argument("--top-k-features", type=int, default=100,
                        help="The number of top features to be considered for plotting. (default: 100).")
    parser.add_argument("--plot-top-k-features", action="store_true", default=False,
                        help="Whether to plot UMAP using the selected top k features. (default: False).")
    parser.add_argument("--min-dist", type=float, default=0.0,
                        help="The effective minimum distance between embedded points for UMAP. (default: 0.0).")
    parser.add_argument("--num-neighbors", type=int, default=5,
                        help="The number of neighboring sample points used for UMAP. (default: 5).")
    parser.add_argument("--perform-cluster", action="store_false", default=True,
                        help="Whether or not to apply clustering. (default: True).")
    parser.add_argument("--cluster-type", type=str, default="kmeans",
                        choices=["kmeans", "gmm", "hdbscan", "spectral", "cocluster", "agglomerative", "affinity"],
                        help="Type of clustering algorithm to be used. (default: kmeans).")
    parser.add_argument("--max-clusters", type=int, default=10,
                        help="The number of clusters to be considered to obtain an optimum number "
                             "of clusters using silhouette score. (default: 10).")
    parser.add_argument("--heatmap-plot", action="store_true", default=False,
                        help="Whether or not to plot heatmap. (default: False).")

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(args=args)


if __name__ == "__main__":
    parse_command_line()
