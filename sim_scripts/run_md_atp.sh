#!/bin/bash
#SBATCH --job-name=md_atp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5-0:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/sita/sim_scripts/logs/md-atp-%j.out


############################
##       Environment      ##
############################
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita

############################
##         Globals        ##
############################
TEMPERATURE=755.95

SCRIPT_DIR="/net/galaxy/home/koes/dap181/labspace/sita/sim_scripts"
MD_SCRIPT="$SCRIPT_DIR/md_simulation.py"

PDB_FILE="/net/galaxy/home/koes/dap181/labspace/sita/data/debug/alanine_tripeptide.pdb"
OUT_DIR="$SCRIPT_DIR/output/atp_${TEMPERATURE}K"
mkdir -p "$OUT_DIR"

############################
##    Launch Simulation   ##
############################
echo "Starting alanine tripeptide MD simulation at ${TEMPERATURE}K..."
python "$MD_SCRIPT" \
    --pdb "$PDB_FILE" \
    --temperature ${TEMPERATURE} \
    --steps 1_000_000_000 \
    --reportInterval 10000 \
    --etrajectory "$OUT_DIR/etrajectory.dcd" \
    --trajectory "$OUT_DIR/trajectory.dcd" \
    --einfo "$OUT_DIR/einfo.log" \
    --info "$OUT_DIR/info.log" \
    --system_pdb "$OUT_DIR/system.pdb"

echo "Simulation COMPLETE."
exit 0
