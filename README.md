<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/logo.svg" width="200">
</p>

### Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics (cell2location model)

[![Docker Repository on Quay](https://quay.io/repository/vitkl/cell2location/status "Docker Repository on Quay")](https://quay.io/repository/vitkl/cell2location)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitkl/cell2location_numpyro/blob/main/docs/notebooks/cell2location_short_demo_colab.ipynb)

We also provide an experimental numpyro translation of the model which has improved memory efficiency (allowing analysis of multiple Visium samples on Google Colab) and minor improvements in speed - https://github.com/vitkl/cell2location_numpyro. You can try it on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitkl/cell2location_numpyro/blob/main/docs/notebooks/cell2location_short_demo_colab.ipynb) - however note that both numpyro itself and cell2location_numpyro are in very active development.

Cell2location maps the spatial distribution of cell types by integrating single-cell RNA-seq (scRNA-seq) and multi-cell spatial transcriptomic data from a given tissue (Fig 1). Cell2location leverages reference cell type signatures that are estimated from scRNA-seq profiles, for example as obtained using conventional clustering to identify cell types and subpopulations followed by estimation of average cluster gene expression profiles. Cell2location implements this estimation step based on Negative Binomial regression, which allows to robustly combine data across technologies and batches. Using these reference signatures, cell2location decomposes mRNA counts in spatial transcriptomic data, thereby estimating the relative and absolute abundance of each cell type at each spatial location (Fig 1). 

Cell2location is implemented as an interpretable hierarchical Bayesian model, (1) providing principled means to account for model uncertainty; (2) accounting for linear dependencies in cell type abundances, (3) modelling differences in measurement sensitivity across technologies, and (4) accounting for unexplained/residual variation by employing a flexible count-based error model. Finally, (5) cell2location is computationally efficient, owing to variational approximate inference and GPU acceleration. For full details and a comparison to existing approaches see our preprint (coming soon). The cell2location software comes with a suite of downstream analysis tools, including the identification of groups of cell types with similar spatial locations.


![Fig1](docs/images/Fig1_v2.png)   
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

Tutorials covering the estimation of expresson signatures of reference cell types (1/3), spatial mapping with cell2location (2/3) and the downstream analysis (3/3) can be found here: https://cell2location.readthedocs.io/en/latest/

Numpyro translation can be used as follows:

```python
from cell2location_numpyro.models.LocationModelLinearDependentWMultiExperimentPyro import LocationModelLinearDependentWMultiExperimentPyro

r = cell2location.run_cell2location(
           model_name=LocationModelLinearDependentWMultiExperimentPyro, ...)
```

There are 2 ways to install and use our package: setup your [own conda environment](https://github.com/BayraktarLab/cell2location#installation-of-dependecies-and-configuring-environment) or use the [singularity](https://github.com/BayraktarLab/cell2location#using-singularity-image) and [docker](https://github.com/BayraktarLab/cell2location#using-docker-image) images (recommended). See below for details.

You can also try cell2location on [Google Colab](https://colab.research.google.com/github/vitkl/cell2location_numpyro/blob/main/docs/notebooks/cell2location_short_demo_colab.ipynb).

Please report bugs and feedback via https://github.com/vitkl/cell2location_numpyro/issues and ask any usage questions in https://github.com/BayraktarLab/cell2location/discussions.

## Configure your own conda environment

1. Installation of dependecies and configuring environment (Method 1 and Method 2)
2. Installation of cell2location

Prior to installing cell2location package you need to install miniconda and create a conda environment containing pymc3 and theano ready for use on GPU. Follow the steps below:

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

#### 1. Method 1: Create conda environment manually

Create conda environment with the required packages pymc3 and scanpy:

```bash
conda create -n cellpymc python=3.7 numpy pandas jupyter leidenalg python-igraph scanpy \
louvain hyperopt loompy cmake nose tornado dill ipython bbknn seaborn matplotlib request \
mkl-service pygpu --channel bioconda --channel conda-forge
```

Do not install pymc3, theano and numpyro with conda because it will not use the system cuda (GPU drivers) and we had problems with cuda installed in the local environment, install them with pip:

```bash
pip install plotnine pymc3>=3.8,<3.10 torch pyro-ppl
pip install -q --upgrade git+https://github.com/pyro-ppl/numpyro.git jax==0.2.3 jaxlib==0.1.56+cuda102 -f
```

#### 1. Method 2: Create environment from file

Create `cellpymc` environment from file, which will install all the required conda and pip packages:

```bash
git clone https://github.com/vitkl/cell2location_numpyro.git
cd cell2location
conda env create -f environment.yml
```

### 2. Install numpyro translation of `cell2location` package

```bash
pip install git+https://github.com/vitkl/cell2location_numpyro.git
```

## Using docker image

Not availlable yet

## Using singularity image

Not availlable yet

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

The architecture of the package is briefly described [here](https://github.com/BayraktarLab/cell2location/blob/master/cell2location/models/README.md). Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Documentation and API details

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez for feedback on the package, Martin Prete for dockerising cell2location and other software support.
