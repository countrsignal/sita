#!/bin/bash
#SBATCH --job-name=IMH_ATP_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-1:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/aita/scripts/logs/atp-imh-eval-12345.out


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

EVAL_SCRIPT="$PROJECT_ROOT/evaluate_imh.py"

cd "$PROJECT_ROOT"

############################
## Launch Training Script ##
############################
echo "Launching IMH evaluation script..."
python "$EVAL_SCRIPT" --seed 12345 --mid_idx 50

echo "IMH evaluation COMPLETE."
exit 0