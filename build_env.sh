#!/bin/bash

# check if micromamba is installed
if ! command -v micromamba &> /dev/null; then
    echo "micromamba could not be found"
    exit 1
else
    echo "micromamba found"
fi

# check if the environment.yaml file exists
if [ ! -f "environment.yaml" ]; then

    echo "environment.yaml file not found"
    exit 1
else
    echo "environment.yaml file found"
fi

set -e # this will stop the script on first error

# load micromamba
eval "$(micromamba shell hook --shell=bash)"

# create micromamba environment
# > automatically answer yes to all questions
micromamba create -y -f environment.yaml

# activate the environment
micromamba activate sita

# install the dependencies
# > micromamba
# > automatically answer yes to all questions
micromamba install -y -c dglteam/label/th24_cu124 -c conda-forge dgl
micromamba install -y lightning -c conda-forge
micromamba install -y -c conda-forge rdkit

# closing message
echo "Done. Run 'micromamba activate sita' to use the environment."