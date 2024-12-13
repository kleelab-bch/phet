# Heterogeneity-Preserving Discriminative Feature Selection for Disease-Specific Subtype Discovery

![Workflow](images/symbol.png)

## Basic Description

This repository encompasses a range of subtype detection algorithms, with a primary focus on the PHet (**P**reserving *
*Het**
erogeneity) algorithm. The PHet algorithm is designed to conduct recurrent subsampling differential analysis and
interquartile
range (IQR) calculations between conditions, in order to pinpoint a minimal set of features that preserve heterogeneity
while maximizing the quality of subtype clustering. Through the utilization of public datasets from microarray and
single-cell RNA-seq studies, PHet has demonstrated its effectiveness in identifying disease subtypes, surpassing the
performance
of previous outlier-based methods. While this guide offers a tutorial on executing 25 different algorithms, it does not
delve
into an exhaustive description of every feature within the package. For additional information about arguments and
functionalities,
it is recommended to execute `python main.py --help`.

## Dependencies

We highly recommend installing [Anaconda](https://www.anaconda.com/) which is an open source distribution of the Python
and R programming languages for data wrangling, predictive analytics, and scientific computing. The codebase is tested
to work under Python 3.11. To install the necessary requirements, run the following commands:

``pip install -r requirements.txt``

Basically, **PHet** requires following packages:

- [numpy](http://www.numpy.org/) (>=1.24)
- [scikit-learn](https://scikit-learn.org/stable/) (>=1.3)
- [pandas](http://pandas.pydata.org/) (>=2.0)
- [scipy](https://www.scipy.org/index.html) (>=1.11)
- [matplotlib](https://matplotlib.org/) (>=3.7)
- [umap-learn](https://github.com/lmcinnes/umap) (>=0.5)
- [statsmodels](https://www.statsmodels.org/stable/index.html) (>=0.14)
- [seaborn](https://seaborn.pydata.org/) (>=0.12)
- [scanpy](https://scanpy.readthedocs.io/en/stable/) (>=1.9)
- [anndata](https://anndata.readthedocs.io/en/latest/) (>=0.9)
- [imblearn](https://imbalanced-learn.org/stable/) (>=0.11)

## Test Samples

Two test datasets with their associated files are provided with this package:

- A microarray [**SRBCT**](https://www.nature.com/articles/nm0601_673/) data:
    - "srbct_matrix.mtx": The the small, round blue-cell tumors expression dataset (83, 2308).
    - "srbct_feature_names.csv": The names of the features of SRBCT data (2308 features).
    - "srbct_classes.csv": Binary classes (0 or 1 ) of samples of the SRBCT data (83 samples).
    - "srbct_types.csv": The subtypes of SRBCT samples (83 samples). Four subtypes of small round blue cell tumors:
      Ewing's sarcoma (EWS), neuroblastoma (NB), rhabdomyosarcoma (RMS), and Burkitt's lymphoma (BL).
    - "srbct_deco_features.csv": Ranked features from DECO on the SRBCT data. Features are ranked based the DECO
      statistics (145, 2).
    - "srbct_limma_features.csv": Results of LIMMA to the SRBCT data. Features are ranked based on the B value, which
      measures the log-odds that a feature is differentially expressed (2308, 7).

- A single cell transcriptomics [**HBECs**](https://www.nature.com/articles/s41586-018-0394-6) data:
    - "hbecs_matrix.mtx": A reduced data from the human bronchial epithelial cells expression dataset (297, 25475).
    - "hbecs_feature_names.csv": The names of the features of HBECs data (25475 features).
    - "hbecs_classes.csv": Binary classes (0 or 1 ) of samples of the HBECs data (297 samples).
    - "hbecs_markers.csv": A predefined list of signatures (411 features).
    - "hbecs_types.csv": The subtypes of HBECs samples (297 samples). Two cell types: Basal and Ionocytes.
    - "hbecs_donors.csv": Three donors for the HBECs data (297 samples).

Please store the files in **one directory** for the best practice.

## Installation and Basic Usage

Run the following commands to clone the repository to an appropriate location:

``git clone https://github.com/kleelab-bch/phet``

For all experiments, navigate to ``src`` folder then run the commands of your choice. For example, to display options
use: `python main.py --help`. It should be self-contained. All the command arguments are initiated
through [main.py](main.py) file. We provided examples on how to run experiments using the SRBCT data.

The description about arguments in the following examples are: *--dspath*: is the location to the dataset folder,
*--rspath*: is the location to the result folder, *--build-syn-dataset*: a true/false variable suggesting whether to
generate simulated data, *--file-name*: is the name of the input data, *--suptitle-name*: is the name of the suptitle of
the figures, *--control-name*: is the name of the control group, *--case-name*: is the name of the case group,
*--methods*: is a list of subtypes detection methods, *--direction*: is the direction of the test the hypothesis test,
*--iqr-range*: is the range where percentiles would be computed on, *--normalize*: type of normalization to be applied,
*--q*: is the percentile to compute,*--dids-scoref*: is the final function to compute features scores for DIDS scoring,
*--num-subsamples*: the number of subsamples, *--feature-weight*: defines weights for binning intervals for PHet,
*--alpha*: is the cutoff significance level, *--score-metric*: is the metric used for evaluation, *--top-k-features*: is
the number of top features to be considered for evaluation and plotting, *--plot-top-k-features*: is the argument to
plot UMAP of the data using top k features, *--cluster-type*: corresponds the the type of clustering algorithm,
*--export-spring*: suggests to export related data for the SPRING plot, and *--num-jobs*: is the number of parallel
workers.

### Example 1

A list of algorithms can be applied at the same time. Here is a simple illustration of how this works:

``
python main.py --dspath [path to the folder containing data] --rspath [path to the folder containing results] --file-name "srbct" --suptitle-name "SRBCT" --control-name "Control" --case-name "Case" --methods ttest_g wilcoxon_g ks_g copa os ort most lsoss dids phet_br --direction "both" --iqr-range 25 75 --normalize "zscore" --q 75 --dids-scoref "tanh" --num-subsamples 1000 --feature-weight 0.4 0.3 0.2 0.1 --alpha 0.01 --score-metric "f1" --top-k-features 100 --cluster-type "kmeans" --num-jobs 2
``

For the *--file-name* argument, please include only the name of the data and remove the suffix *_matrix.mtx*. This will
generate several files located in the **rspath** folder.

### Example 2

To infer subtypes using LIMMA and DECO. First, you need to
run [LIMMA](https://bioconductor.org/packages/release/bioc/html/limma.html)
and [DECO](https://bioconductor.org/packages/release/bioc/html/deco.html) then store the features in *.csv* format with
appropriate suffixes. Here, we show an example of how to get subtypes using
features ([srbct_limma_features.csv](samples/srbct_limma_features.csv) & [srbct_deco_features.csv](samples/srbct_deco_features.csv))
from these algorithms:

``
python main.py --dspath [path to the folder containing data] --rspath [path to the folder containing results] --file-name "srbct" --suptitle-name "SRBCT" --control-name "Control" --case-name "Case" --methods limma_g deco --alpha 0.01 --score-metric "f1" --top-k-features 100 --cluster-type "kmeans" --num-jobs 2
``

For the *--file-name* argument, please include only the name of the data and remove the suffix *_matrix.mtx*. This will
generate several files located in the **rspath** folder.

### Example 3

To export file for the SPRING plot, enable the argument *--export-spring*. Here, we run the **PHet** (with IQR) model
using the [HBECs](samples/plasschaert_human_basal_vs_ionocytes_matrix.mtx) data:

``
python main.py --dspath [path to the folder containing data] --rspath [path to the folder containing results] --file-name "hbecs" --suptitle-name "Basal vs Ionocytes" --control-name "Basal" --case-name "Ionocytes" --methods phet_br --export-spring --iqr-range 25 75 --normalize "zscore" --num-subsamples 1000 --feature-weight 0.4 0.3 0.2 0.1 --alpha 0.01 --score-metric "f1" --top-k-features 100 --cluster-type "kmeans" --num-jobs 2
``

For the *--file-name* argument, please include only the name of the data and remove the suffix *_matrix.mtx*. This will
generate several files located in the **rspath** folder.

## Integrating PHet with SCANPY and Seurat

To effectively integrate PHet with established frameworks such as SCANPY and Seurat, the procedure encompasses two primary steps. First, execute PHet to generate the requisite features from your dataset. Thereafter, import these features into SCANPY or Seurat for subsequent analysis. The following outlines these two essential steps:

### Step 1: Run PHet (using the sample data "SRBCT")

``
python main.py --dspath [path to the folder containing data] --rspath [path to the folder containing results] --file-name "srbct" --suptitle-name "SRBCT" --control-name "Control" --case-name "Case" --methods ttest_g wilcoxon_g ks_g copa os ort most lsoss dids phet_br --direction "both" --iqr-range 25 75 --normalize "zscore" --q 75 --dids-scoref "tanh" --num-subsamples 1000 --feature-weight 0.4 0.3 0.2 0.1 --alpha 0.01 --score-metric "f1" --top-k-features 100 --cluster-type "kmeans" --num-jobs 2
``

For the *--file-name* argument, please include only the name of the data and remove the suffix *_matrix.mtx*. This will
generate several files located in the **rspath** folder.

### Step 2a: Import features into SCANPY

``` python
# Load the scanpy package
import scanpy as sc
# Load the full feature set from the SRBCT dataset
full_features = pd.read_csv(os.path.join([path to the folder containing data],
                                         "srbct_feature_names.csv"), sep=',')
full_features = full_features["features"].to_list()
full_features = [f.upper() for f in full_features]
# Incorporate PHet's features
phet_features = pd.read_csv(os.path.join([path to the folder containing results],
                                          "srbct_phet_br_features.csv"),
                            sep=',', header=None)
phet_features = np.squeeze(phet_features.values.tolist())
phet_features = [f.upper() for f in phet_features]
# Load the corresponding expression values for the SRBCT dataset
adata = sc.read_mtx(os.path.join([path to the folder containing data], "srbct_matrix.mtx"))
adata.var_names = full_features
# Reduce the feature size of the SRBCT dataset by including only PHet's features
adata = adata[:, phet_features]
```

### Step 2b: Import features into Seurat

``` R
# Load R required packages
require(Seurat)
require(Matrix)
# Load the full feature set from the SRBCT dataset
full_features <- read.csv(file.path([path to the folder containing data], 
                                    "srbct_feature_names.csv"), 
                          header = TRUE)
full_features <- full_features$features
# Incorporate PHet's features
phet_features <- read.csv(file.path([path to the folder containing results], 
                                    "srbct_phet_br_features.csv"), 
                          header = FALSE)
phet_features <- phet_features$V1
# Load the corresponding expression values for the SRBCT dataset
X <- readMM(file = file.path(data_dir, "srbct_matrix.mtx"))
colnames(X) <- full_features
# Reduce the feature size of the SRBCT dataset by including only PHet's features
X <- X[, phet_features]
seurat_obj <- CreateSeuratObject(counts = t(X))
```

## Citing

If you find **PHet** useful in your research, please consider citing the following paper:

- M. A. Basher, Abdur Rahman, Hallinan, Caleb, and Lee, Kwonmoo. *
  *["Heterogeneity-Preserving Discriminative Feature Selection for Disease-Specific Subtype Discovery."](https://doi.org/10.1101/2023.05.14.540686)
  **, bioRxiv (2024).

## Contact

For any inquiries, please contact: [ar.basher@childrens.harvard.edu](mailto:ar.basher@childrens.harvard.edu)