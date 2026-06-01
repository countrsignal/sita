#!/bin/bash
#SBATCH --job-name=md_adp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-0:00:00
#SBATCH --partition=koes_gpu
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dap181@pitt.edu
#SBATCH --output=/net/pulsar/home/koes/dap181/labspace/sita/sim_scripts/logs/md-adp-%j.out


############################
##       Environment      ##
############################
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita

############################
##         Globals        ##
############################
SCRIPT_DIR="/net/galaxy/home/koes/dap181/labspace/sita/sim_scripts"
MD_SCRIPT="$SCRIPT_DIR/md_simulation.py"

PDB_FILE="/net/galaxy/home/koes/dap181/labspace/sita/data/debug/alanine_dipeptide.pdb"
OUT_DIR="$SCRIPT_DIR/output/adp"
mkdir -p "$OUT_DIR"

############################
##    Launch Simulation   ##
############################
echo "Starting alanine dipeptide MD simulation..."
python "$MD_SCRIPT" \
    --pdb "$PDB_FILE" \
    --temperature 300 \
    --steps 1000000000 \
    --reportInterval 10000 \
    --etrajectory "$OUT_DIR/etrajectory.dcd" \
    --trajectory "$OUT_DIR/trajectory.dcd" \
    --einfo "$OUT_DIR/einfo.log" \
    --info "$OUT_DIR/info.log" \
    --system_pdb "$OUT_DIR/system.pdb"

echo "Simulation COMPLETE."
exit 0
