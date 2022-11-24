# DIGRAC
DIGRAC: Directed Graph Clustering Based on Flow Imbalance (LoG 2022)

For details, please read [our paper](https://arxiv.org/pdf/2106.05194.pdf).

**Citing**


If you find DIGRAC useful in your research, please consider adding the following citation:

```bibtex
@article{he2021digrac,
  title={DIGRAC: Digraph Clustering Based on Flow Imbalance},
  author={He, Yixuan and Reinert, Gesine and Cucuringu, Mihai},
  journal={arXiv preprint arXiv:2106.05194},
  year={2021}
}
```

--------------------------------------------------------------------------------

## Environment Setup
### Overview
<!-- The underlying project environment composes of following componenets: -->
The project has been tested on the following environment specification:
1. Ubuntu 18.04.5 LTS (Other x86_64 based Linux distributions should also be fine, such as Fedora 32)
2. Nvidia Graphic Card (NVIDIA GeForce RTX 2080 with driver version 440.36, and NVIDIA RTX 8000) and CPU (Intel Core i7-10700 CPU @ 2.90GHz)
3. Python 3.6.13 (and Python 3.6.12)
4. CUDA 10.2 (and CUDA 9.2)
5. Pytorch 1.8.0 (built against CUDA 10.2) and Pytorch 1.6.0 (build against CUDA 9.2 and CUDA 11.2)
6. Other libraries and python packages (See below)

### Installation method 1 (.yml files)
You should handle (1),(2) yourself. For (3), (4), (5) and (6), we provide a list of steps to install them.

<!-- We place those python packages that can be easily installed with one-line command in the requirement file for `pip` (`requirements_pip.txt`). For all other python packages, which are not so well maintained by [PyPI](https://pypi.org/), and all C/C++ libraries, we place in the conda requirement file (`requirements_conda.txt`). Therefore, you need to run both conda and pip to get necessary dependencies. -->

We provide two examples of envionmental setup, one with CUDA 10.2 and GPU, the other with CPU.

Following steps assume you've done with (1) and (2).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Both Miniconda and Anaconda are OK.

2. Create an environment and install python packages (GPU):
```
conda env create -f environment_GPU.yml
```

3. Create an environment and install python packages (CPU):
```
conda env create -f environment_CPU.yml
```


### Installation method 2 (manual installation)
The codebase is implemented in Python 3.6.12. package versions used for development are below.
```
networkx           2.5
tqdm               4.50.2
numpy              1.19.2
pandas             1.1.4
texttable          1.6.3
latextable         0.1.1
scipy              1.5.4
argparse           1.1.0
sklearn            0.23.2
stellargraph       1.2.1 (for link direction prediction: conda install -c stellargraph stellargraph)
torch              1.8.0
torch-scatter      2.0.5
torch-geometric    1.6.3 (follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
matplotlib         3.3.4 (for generating plots and results)
```

### Execution checks
When installation is done, you could check you enviroment via:
```
cd execution
bash setup_test.sh
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use parallel (https://www.gnu.org/software/parallel/, can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets.

- ./src/ stores files to train various models, utils and metrics.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./logs/ stores trained models and logs, as well as predicted clusters (optional). When you are in debug mode (see below), your logs will be stored in ./debug_logs/ folder.

## Options
<p align="justify">
DIGRAC provides the following command line arguments, which can be viewed in the ./src/param_parser.py.
</p>

### Synthetic data options:
See file ./src/param_parser.py.

```
  --p                     FLOAT         Probability of the existence of a link.                 Default is 0.02. 
  --eta                   FLOAT         Probability of flipping the direction of each edge.     Default is 0.1.
  --N                     INT           (Expected) Number of nodes in an DSBM.                  Default is 1000.
  --K                     INT           Number of clusters/blocks in an DSBM.                   Default is 3.
```

### Major model options:
See file ./src/param_parser.py.

```
  --epochs                INT         Number of DIGRAC (maximum) training epochs.               Default is 1000. 
  --early_stopping        INT         Number of DIGRAC early stopping epochs.                   Default is 200. 
  --num_trials            INT         Number of trials to generate results.                     Default is 10.
  --seed_ratio            FLOAT       Ratio in the training set of each cluster 
                                                        to serve as seed nodes.                 Default is 0.
  --train_ratio           FLOAT       Training ratio.                                           Default is 0.8.  
  --test_ratio            FLOAT       Test ratio.                                               Default is 0.1.
  --lr                    FLOAT       Initial learning rate.                                    Default is 0.01.  
  --weight_decay          FLOAT       Weight decay (L2 loss on parameters).                     Default is 5^-4. 
  --dropout               FLOAT       Dropout rate (1 - keep probability).                      Default is 0.5.
  --hidden                INT         Number of hidden units.                                   Default is 32. 
  --seed                  INT         Random seed.                                              Default is 31.
  --no-cuda               BOOL        Disables CUDA training.                                   Default is False.
  --debug, -D             BOOL        Debug with minimal training setting, not to get results.  Default is False.
  --regenerate_data       BOOL        Whether to force creation of data splits.                 Default is False.
  --load_only             BOOL        Whether not to store generated data.                      Default is False.
  -AllTrain, -All         BOOL        Whether to use all data to do gradient descent.           Default is False.
  --SavePred, -SP         BOOL        Whether to save predicted labels.                         Default is False.
  --dataset               STR         Data set to consider.                                     Default is 'DSBM/'.
  --F_style               STR         Meta-graph adjacency matrix style.                        Default is 'cyclic'.
  --normalizations        LST         Normalization methods to choose from: 
                                        vol_min, vol_sum, vol_max and None.                     Default is ['vol_sum'].
  --thresholds            LST         Thresholding methods to choose from: sort, std and None.  Default is ['sort'].
```


## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce DIGRAC results.
```
bash DIGRAC_jobs.sh
```
Other execution files are similar to run.

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files.

You can also use CPU for training if you add ``--no-duca", or GPU if you delete this.

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```

Then, below are various options to try:

Creating an DIGRAC model for DSBM of the default setting.
```
python ./train.py
```
Creating an DIGRAC model for DSBMs with 5000 nodes, ``complete" meta-graph structure and p=0.1.
```
python ./train.py --N 5000 --F_style complete --p 0.1
```
Creating a model for Telegram data set with some custom learning rate and epoch number, save the predicted clusters and use all data for training.
```
python ./train.py --dataset telegram --lr 0.001 --epochs 300 -SP -All
```
Creating a model for blog data set with specific number of trials and use CPU.
```
python ./train.py --dataset blog -SP -All --no-cuda --num_trials 5
```

## Extra results and environment requirements

- Running scripts: execution/extra_jobs.sh, calling src/extra_train.py with the extra file src/extra_comparison.py

- Methods to compare against: InfoMap from ```pip install infomap```, Leiden ad Louvain to maximize modularity from https://github.com/vtraag/leidenalg, also OSLOM from https://github.com/hhromic/python-oslom-runner.

- Package requirements: If one wants to run these extra experiements on non-spectral and non-GNN methods, then extra packages are required. Please follow the following links to fulfill the requirements: https://mapequation.github.io/infomap/python/, https://github.com/vtraag/leidenalg, and https://github.com/hhromic/python-oslom-runner.

- To get OSLOM run in Python 3.6 or higher version, we need to fix a line in the package installation. 
In PATH-TO-PYTHON/site-packages/oslom/runner.py in read_clusters(self, min_cluster_size)
```
    121         with open(self.get_path(OslomRunner.OUTPUT_FILE), "r") as reader:
    122             # Read the output file every two lines
--> 123             for line1, line2 in itertools.izip_longest(*[reader] * 2):
    124                 info = OslomRunner.RE_INFOLINE.match(line1.strip()).groups()
    125                 nodes = line2.strip().split(" ")
```
We need to change ```izip_longest``` to ```zip_longest```.

- Results are saved in extra_results folder.