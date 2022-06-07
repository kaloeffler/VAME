# VAME for the THEMIS Data

Adaption of the [VAME](https://github.com/LINCellularNeuroscience/VAME/blob/master/README.md) to apply it on the THEMIS data with additional functionalities to explore the learned latent space.

# VAME in a Nutshell
VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/) based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.

The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## Installation
To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. 
Here, you can create a [virtual enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to store all the dependencies necessary for VAME. (you can also use the VAME.yaml file supplied here, byt simply openning the terminal, running `git clone https://github.com/kaloeffler/VAME`, then type `cd VAME` then run: `conda env create -f VAME.yaml`). Then activate the conda environment `conda activate venv_VAME`.

* Go to the locally cloned VAME directory and run `python setup.py install` in order to install VAME in your active conda environment.
* (Install the current stable Pytorch release using the OS-dependent instructions from the [Pytorch website](https://pytorch.org/get-started/locally/). Currently, VAME is tested on PyTorch 1.5. (Note, if you use the conda file we supply, PyTorch is already installed and you don't need to do this step.))

## Workflow
See the `themis_pipeline.py` to get started, which includes all steps from data preprocessing over model training to prediction of latent vectors. Then explore the latent space using the *.inbpy scripts in `analysis_scripts/`

### Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer. Adaptions of the VAME code to work on the THEMIS data including additional scripts for visualizing the latent space have been developed by Katharina Löffler.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/).
As such, the VAME project management codebase has been adapted from the DeepLabCut codebase.
The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](http:\\deeplabcut.org), released under LGPL v3.0.
The implementation of the VRAE model is partially adapted from the [Timeseries clustering](https://github.com/tejaslodaya/timeseries-clustering-vae) repository developed by [Tejas Lodaya](https://tejaslodaya.com).

### References
VAME preprint: [Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2) <br/>
Kingma & Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) <br/>
Pereira & Silveira: [Learning Representations from Healthcare Time Series Data for Unsupervised Anomaly Detection](https://www.joao-pereira.pt/publications/accepted_version_BigComp19.pdf)

### License: GPLv3
See the [LICENSE file](../master/LICENSE) for the full statement.
