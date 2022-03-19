#!/bin/bash
export CONDA_ALWAYS_YES="true"

conda create -n gsoc_env python=3.7
conda activate gsoc_env
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge matplotlib
conda install -c anaconda   scikit-learn \
                            scikit-image
conda install pandas

# install pip packages
pip install vit_pytorch \
            split-folders \
            e2cnn \
            torchinfo \
            einops

!pip install --upgrade --no-cache-dir gdown

# installing apt-get packages
apt-get update
apt-get -y install unzip
