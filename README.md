# aita

## Prerequisites

- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (the build script uses the `micromamba` CLI)
- For the SLURM batch scripts (`sim_scripts/run_md_adp.sh`, `scripts/run_jump_anneal_adp.sh`, and similar): a cluster with SLURM and GPU support as configured in each script’s `#SBATCH` lines, plus `micromamba activate sita`

## Build the environment with `build_env.sh`

From the root of this repository:

1. Ensure micromamba is installed and on your `PATH`.
2. Run the build script (it stops on the first error):

   ```bash
   cd /path/to/aita
   bash build_env.sh
   ```

What `build_env.sh` does:

1. Loads the micromamba shell hook for bash.
2. Creates the conda environment defined in `environment.yaml` (the environment is named **`sita`** in that file).
3. Activates `sita` and installs additional packages: DGL (CUDA 12.4 build), PyTorch Lightning, and RDKit from the channels specified in the script.

After a successful run, activate the same environment in new shells:

```bash
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita
```

**Note:** Creating the environment requires network access to conda channels and pip (including the `bgflow` git dependency in `environment.yaml`). If any step fails, fix the reported error and re-run; you may need to remove a partially created env (`micromamba env remove -n sita`) before retrying.

## Example: MD data generation with `sim_scripts/run_md_adp.sh`

`sim_scripts/run_md_adp.sh` is a SLURM batch job that runs OpenMM molecular dynamics for alanine dipeptide (ADP), producing trajectories and logs under `sim_scripts/output/adp/`. That kind of simulation output is used as input for downstream training experiments in this project.

### 1. Align paths and scheduler settings with your machine

The script may still reference another checkout or cluster paths (for example `.../sita/...` and `#SBATCH` options). Before submitting, edit `sim_scripts/run_md_adp.sh` so that at least these match your setup:

- `#SBATCH` lines: `partition`, `mail-user`, `output` (log directory must exist or be creatable), memory, time limit, and GPU request as allowed on your cluster.
- `SCRIPT_DIR`: absolute path to **this** repo’s `sim_scripts` directory.
- `PDB_FILE`: path to the ADP structure, e.g. `data/debug/alanine_dipeptide.pdb` in this repository.
- `OUT_DIR`: where you want trajectories and logs written (the script creates the directory if needed).

Ensure the log directory in `#SBATCH --output=...` exists (e.g. `sim_scripts/logs`) if your scheduler does not create it.

### 2. Submit the job

With the edits saved and your account configured for SLURM:

```bash
cd /path/to/aita
sbatch sim_scripts/run_md_adp.sh
```

The job activates the **`sita`** micromamba environment (same name as in `build_env.sh` / `environment.yaml`), then runs `md_simulation.py` with the arguments in the script (temperature, step count, report interval, output file names).

### 3. Running without SLURM (optional smoke test)

To verify OpenMM and the environment on a single machine with a GPU, you can run the Python driver directly after `micromamba activate sita`, pointing paths at this repo:

```bash
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita
cd /path/to/aita

OUT_DIR="sim_scripts/output/adp_test"
mkdir -p "$OUT_DIR"

python sim_scripts/md_simulation.py \
  --pdb data/debug/alanine_dipeptide.pdb \
  --temperature 300 \
  --steps 1000000 \
  --reportInterval 10000 \
  --etrajectory "$OUT_DIR/etrajectory.dcd" \
  --trajectory "$OUT_DIR/trajectory.dcd" \
  --einfo "$OUT_DIR/einfo.log" \
  --info "$OUT_DIR/info.log" \
  --system_pdb "$OUT_DIR/system.pdb"
```

Use a smaller `--steps` value than the billion-step production setting in `run_md_adp.sh` for quick tests; increase it when you intend to match published training data volumes.

## Bootstrap training: `scripts/run_jump_anneal_adp.sh`

This job runs the jump-anneal bootstrap trainer for the ADP “jump prior” setup via Hydra: it executes `jump_anneal.py` with `experiment=anneal_adp_jump_prior`, plus `trainer.max_epochs=200` and `loader.batch_size=512` (you can override these on the command line if you run Python directly).

### 1. Checkpoints and config

The experiment is defined in `configs/experiment/anneal_adp_jump_prior.yaml`. It points at pretrained EBM and flow checkpoints under `${paths.root_dir}/experimental_results/...`. Ensure those paths exist on your machine, or edit the YAML (or pass Hydra overrides) so `ebm_model_ckpt` and `flow_model_ckpt` resolve to your own checkpoint files.

### 2. Weights & Biases

The shell script writes a `.env` file in the project root with `WANDB_ENTITY` set from the `USERNAME` variable inside the script (currently a placeholder team or user name). Change `USERNAME` in `scripts/run_jump_anneal_adp.sh` to your W&B entity before submitting, or remove or adjust the `.env` logic if you configure logging another way.

### 3. Align SLURM and log paths

Edit `#SBATCH` options in `scripts/run_jump_anneal_adp.sh` for your site: `partition`, `mail-user`, `output` (create `scripts/logs` or another log directory if required), memory, time limit, GPU count, `exclude` nodes, and `ntasks-per-node` as appropriate.

### 4. Submit from the repository root

The script sets `PROJECT_ROOT` from `SLURM_SUBMIT_DIR` when running under SLURM, so the working directory used at submit time must be the **aita** repository root (where `jump_anneal.py` lives):

```bash
cd /path/to/aita
sbatch scripts/run_jump_anneal_adp.sh
```

The job activates **`sita`**, writes `.env`, `cd`s to `PROJECT_ROOT`, and runs the `python jump_anneal.py ...` command embedded in the script.

### 5. Running without SLURM (interactive or single GPU)

From the repo root with the same environment:

```bash
eval "$(micromamba shell hook --shell=bash)"
micromamba activate sita
cd /path/to/aita

echo "WANDB_ENTITY='your_wandb_entity'" > .env

python jump_anneal.py experiment=anneal_adp_jump_prior \
  trainer.max_epochs=200 \
  loader.batch_size=512
```

You can append further Hydra overrides (for example different `trainer.max_epochs` or checkpoint paths) on the same command line.
