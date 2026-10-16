#!/bin/bash
#SBATCH --job-name=transchan
#SBATCH --array=0-19%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=transchan-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
"${TRANSCHAN_PYTHON:-python}" -u -m active_learning_transchan run --task-id "${SLURM_ARRAY_TASK_ID}" "$@"
