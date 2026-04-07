#!/bin/bash
#SBATCH --job-name=AITA_pretrain_atflow_adp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=g016,g017,g018
#SBATCH --time=6-0:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/aita/scripts/logs/aita-pretrain-atflow-adp-600epochs-0lnb.out


############################
##       Environment      ##
############################
eval "$(micromamba shell hook --shell=bash)"
micromamba activate aita


############################
##         Globals        ##
############################
USERNAME="countrsignal"

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR")
else
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
fi

PRETRAIN_SCRIPT="$PROJECT_ROOT/pretrain.py"

cd "$PROJECT_ROOT"

############################
##    Create .env file    ##
############################
echo "WANDB_ENTITY='${USERNAME}'" > .env


############################
## Launch Training Script ##
############################
echo "Launching training script..."
# python "$PRETRAIN_SCRIPT" experiment=pretrain_flow_adp_temp trainer.max_epochs=20 loader.batch_size=512
# python "$PRETRAIN_SCRIPT" experiment=pretrain_flow_atp_temp trainer.max_epochs=500 loader.batch_size=400
python "$PRETRAIN_SCRIPT" experiment=pretrain_atflow_adp trainer.max_epochs=600 loader.batch_size=512

echo "Training COMPLETE."
exit 0