#!/bin/bash
set -e # this will stop the script on first error

# load micromamba
eval "$(micromamba shell hook --shell=bash)"

# create micromamba environment
micromamba create -f environment.yaml

# activate the environment
micromamba activate aita

# install the dependencies
# > micromamba
micromamba install -c dglteam/label/th24_cu124 -c conda-forge dgl
micromamba install lightning -c conda-forge
