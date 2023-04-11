__author__ = "Abdurrahman Abul-Basher"
__date__ = '02/06/2021'
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
    if sys.platform.startswith('win'):
        os.system("cls")
    else:
        os.system("clear")
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__, "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45, subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    args = Arguments()

    ###***************************         Global arguments         ***************************###
    args.num_jobs = parse_args.num_jobs
    args.export_spring = parse_args.export_spring

    ###***************************          Path arguments          ***************************###
    args.dspath = parse_args.dspath
    args.rspath = parse_args.rspath

    ###***************************          File arguments          ***************************###
    args.file_name = parse_args.file_name

    ###***************************     Preprocessing arguments      ***************************###
    args.suptitle_name = parse_args.suptitle_name

    ###***************************       Inference arguments        ***************************###
    args.q = parse_args.q
    args.iqr_range = parse_args.iqr_range
    args.direction = parse_args.direction
    args.permutation_test = parse_args.permutation_test
    # for OutlierSumStatistic
    args.two_sided_test = parse_args.two_sided_test
    # for MOST
    args.k = parse_args.k
    # for DIDS
    args.dids_scoref = parse_args.dids_scoref
    # for DeltaIQR and PHet
    args.normalize = parse_args.normalize
    # for PHet
    args.num_subsamples = parse_args.num_subsamples
    args.calculate_deltaiqr = parse_args.calculate_deltaiqr
    args.calculate_fisher = parse_args.calculate_fisher
    args.calculate_profile = parse_args.calculate_profile
    args.bin_KS_pvalues = parse_args.bin_KS_pvalues
    args.feature_weight = parse_args.feature_weight
    args.weight_range = parse_args.weight_range

    ###***************************        Evaluate arguments        ***************************###
    args.sort_by_pvalue = parse_args.sort_by_pvalue
    args.pvalue = parse_args.pvalue

    ###***************************    UMAP & Clustering arguments   ***************************###
    args.standardize = parse_args.standardize
    args.num_neighbors = parse_args.num_neighbors
    args.min_dist = parse_args.min_dist
    args.perform_cluster = parse_args.perform_cluster
    args.cluster_type = parse_args.cluster_type
    args.apply_hungarian = parse_args.apply_hungarian
    args.heatmap_plot = parse_args.heatmap_plot

    return args


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run subtypes detection algorithms.")

    # Global arguments
    parser.add_argument('--methods', nargs="+", type=str, default=["ttest", "copa", "oss", "ort", "most",
                                                                   "lsoss", "dids", "deltaiqr", "phet"],
                        help='Select subtype methods. (default: ["ttest", "copa", "oss", "ort", "most", "lsoss", "dids", "deltaiqr", "phet"]).')
    parser.add_argument('--num-jobs', type=int, default=2,
                        help='Number of parallel workers. (default: 2).')
    parser.add_argument('--export-spring', action='store_true', default=False,
                        help='Whether or not to export data for the SPRING plot. (default: False).')

    # Arguments for file paths
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='Path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='Path to the results. The default is set to result '
                             'folder outside the source code.')

    # Arguments for file names and models
    parser.add_argument('--file-name', type=str, required=True,
                        help='The file name to save an object.')

    # Arguments for preprocessing dataset
    parser.add_argument('--suptitle-name', type=str, default='temp',
                        help='The name of the suptitle of the figures. (default: "temp")')

    # Arguments for inference
    parser.add_argument('--q', type=float, default=75.0,
                        help='Percentile to compute, which must be between 0 and 100 inclusive. '
                             '(default: 75.0).')
    parser.add_argument('--iqr-range', action='store_true', default=False,
                        help='Two-element sequence containing floats in range of [0,100]. '
                             'Percentiles over which to compute the range. Each must be '
                             'between 0 and 100, inclusive. (default: (25, 75)).')
    parser.add_argument('--direction', type=str, default='both',
                        help='Direction to compute the nonparametric permutation test. '
                             'Possible values are: up, down, and both. (default: up).')
    parser.add_argument('--permutation-test', action='store_true', default=False,
                        help='Whether or not to compute the nonparametric permutation test. '
                             '(default value: False).')
    ## for OutlierSumStatistic
    parser.add_argument('--two-sided-test', action='store_false', default=True,
                        help='Whether or not to compute the two sided test for OutlierSumStatistic. '
                             '(default: True).')
    ## for MOST
    parser.add_argument("--k", type=int, default=None,
                        help="The k number of case samples to compute for MOST. "
                             "(default: None).")
    ## for DIDS
    parser.add_argument('--dids-scoref', type=str, default='tanh',
                        help='DIDS scoring function. Possible values are: tanh, sqrt, and quad. '
                             '(default: tanh).')
    ## for DeltaIQR and PHet
    parser.add_argument('--normalize', type=str, default='zscore',
                        help='Type of normalization. Possible values are: zscore and robust. '
                             '(default: zscore).')
    ## for PHet
    parser.add_argument('--num-subsamples', type=int, default=1000,
                        help='The number of subsamples for PHet. '
                             '(default: 1000).')
    parser.add_argument('--calculate-deltaiqr', action='store_false', default=True,
                        help='Whether or not to compute IQR differences between two samples for PHet. '
                             '(default: True).')
    parser.add_argument('--calculate-fisher', action='store_false', default=True,
                        help='Whether or not to compute Fisher\'s method for PHet. '
                             '(default: True).')
    parser.add_argument("--calculate-profile", action='store_false', default=True,
                        help='Whether or not to compute features profiles for PHet. '
                             '(default: True).')
    parser.add_argument('--bin-KS-pvalues', action='store_false', default=True,
                        help='Whether to use binning strategy for PHet. '
                             '(default: True).')
    parser.add_argument('--feature-weight', nargs="+", type=float,
                        default=[0.4, 0.3, 0.2, 0.1],
                        help="Four hyper-parameters for PHet. "
                             "(default: [0.4, 0.3, 0.2, 0.1]).")
    parser.add_argument('--weight-range', nargs="+", type=float,
                        default=[0.1, 0.4, 0.8],
                        help="Six hyper-parameters for constraints. "
                             "(default: [0.1, 0.4, 0.8]).")

    # Arguments for evaluation
    parser.add_argument('--sort-by-pvalue', action='store_false', default=True,
                        help='Whether or not to sort scores by pvalue using Gamma distribution. '
                             '(default: True).')
    parser.add_argument('--pvalue', action='store_false', default=True,
                        help='Significance level. (default: 0.01).')

    # Arguments for UMAP and clustering
    parser.add_argument('--standardize', type=int, default=200,
                        help='Top k labels to be considered for predicting. Only considered when'
                             ' the prediction strategy is set to "pref-rank" option. (default: 200).')
    parser.add_argument('--num_neighbors', type=int, default=10,
                        help='The size of local neighborhood (e.g., number of neighboring samples) used for UMAP.'
                             '(default: 10).')
    parser.add_argument('--min-dist', type=float, default=0.0,
                        help='The effective minimum distance between embedded points for UMAP. (default: 0.0).')
    parser.add_argument('--perform-cluster', action='store_false', default=True,
                        help='Whether or not to apply clustering. (default: True).')
    parser.add_argument('--cluster-type', type=str, default="spectral",
                        help='Type of clustering. Possible values are: kmeans, gmm, hdbscan, spectral, '
                             'cocluster, agglomerative, and affinity. (default: spectral).')
    parser.add_argument('--apply-hungarian', action='store_true', default=False,
                        help='Whether or not to apply the Hungarian algorithm to assign each point to a true label. '
                             '(default: False).')
    parser.add_argument('--heatmap-plot', action='store_true', default=False,
                        help='Whether or not to plot heatmap. (default: False).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(args=args)


if __name__ == "__main__":
    parse_command_line()
