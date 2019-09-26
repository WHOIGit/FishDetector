#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=output_%x.out

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

module load cuda91/toolkit cuda91/cudnn darknet

# If there is a local build of darknet, prefer it
if [ -e ./darknet ]; then
	PATH="$(pwd)/darknet:$PATH"
fi

darknet detector train obj.data yolo-obj.cfg darknet53.conv.74 -gpus "$CUDA_VISIBLE_DEVICES"

date

