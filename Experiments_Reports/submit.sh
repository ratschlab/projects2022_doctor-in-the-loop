#!/bin/bash

#SBATCH --tmp=100G
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=1
#SBATCH --time=20:00:00


mkdir ${TMPDIR}/sites
mkdir ${TMPDIR}/reforestree
mkdir ${TMPDIR}/patches
rsync -raq /cluster/scratch/vbarenne/data/dataset/sites/* ${TMPDIR}/sites/.
rsync -raq /cluster/scratch/vbarenne/data/dataset/patches/* ${TMPDIR}/patches/.
rsync -raq /cluster/scratch/vbarenne/data/dataset/patches_df.csv ${TMPDIR}/patches_df.csv
rsync -raq /cluster/work/igp_psr/ai4good/group-3b/reforestree/field_data.csv ${TMPDIR}/reforestree/.
python3 main.py 
