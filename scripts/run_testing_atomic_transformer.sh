#!/bin/bash
#SBATCH --job-name=SITA_test_atomic_tfm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=g016,g017,g018,g019,g020
#SBATCH --time=2-0:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/sita/scripts/logs/sita-test-atomic-transformer.out


############################
##       Environment      ##
############################
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita


############################
##         Globals        ##
############################
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR")
else
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
fi

TEST_SCRIPT="$PROJECT_ROOT/scripts/testing_atomic_transformer.py"

cd "$PROJECT_ROOT"


############################
##    Launch Test Script  ##
############################
echo "Launching testing script..."
python "$TEST_SCRIPT"

echo "Testing COMPLETE."
exit 0
