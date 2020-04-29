#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output_%x.out

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date
trap date EXIT

module load cuda91/toolkit cuda91/cudnn

BASEDIR="$(pwd)"

# Use our local build of Darknet
PATH="$BASEDIR/darknet/build_release:$PATH"

# Make a directory for this job
JOB_DIR=training/$(date +'%Y-%m-%d')_$SLURM_JOB_ID
mkdir -p "$JOB_DIR"
cd "$JOB_DIR"

# Create config file
python configtool.py \
    --classes 1 \
    --batch 64 \
    --subdivisions 8 \
    --no-color-adjustments \
    --size 416 960 \
    "$BASEDIR"/darknet/cfg/yolov4-custom.cfg \
    > yolo-obj.cfg

print "Starting single GPU training"
darknet detector train \
    "$BASEDIR"/obj.data \
    ./yolo-obj.cfg \
    "$BASEDIR"/pretrained/yolov4.conv.137 \
    -gpus 0
