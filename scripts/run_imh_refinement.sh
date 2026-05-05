#!/bin/bash
#SBATCH --job-name=IMH_ATP
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=g016,g017,g018,g019
#SBATCH --time=2-0:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/aita/scripts/logs/atp-imh-refinement-03.out


############################
##       Environment      ##
############################
eval "$(micromamba shell hook --shell=bash)"
micromamba activate aita

############################
##         Globals        ##
############################

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR")
else
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
fi

MCMC_SCRIPT="$PROJECT_ROOT/atp_imh_refinement.py"

cd "$PROJECT_ROOT"

############################
## Launch Training Script ##
############################
echo "Launching IMH refinement script..."
python "$MCMC_SCRIPT" --seed 12345

echo "IMH refinement COMPLETE."
exit 0