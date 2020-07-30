#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.log

set -eu

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date
trap date EXIT

module load cuda10.1/{toolkit,blas,fft}
module load gcc/6.5.0

BASEDIR="$(pwd)"

# Use our local build of Darknet
PATH="$BASEDIR/darknet/build_release:$PATH"

# Get configuration options
DATADIR="$(cd "$1"; pwd)"

# Make a directory for this job
JOB_DIR=networks/"$(date +'%Y-%m-%d')"_"$SLURM_JOB_ID"
mkdir -p "$JOB_DIR"
cd "$JOB_DIR"
ln -s "$BASEDIR"/logs/train_"$SLURM_JOB_ID".log train.log
ln -s "$DATADIR" data

# Generate the training config files
cp "$BASEDIR"/obj.names .
CLASSES="$(wc -l ./obj.names | cut -d ' ' -f 1)"

for LIST in data/{test,train}.txt; do
    sed -e 's,^,data/,g' "$LIST" > "$(basename "$LIST")"
done

python "$BASEDIR"/configtool.py \
    --classes "$CLASSES" \
    --batch 64 \
    --subdivisions 32 \
    --no-color-adjustments \
    --size 416 960 \
    "$BASEDIR"/darknet/cfg/yolov4-custom.cfg \
    > yolo-obj.cfg

# Generate obj.data
cat <<EOF > obj.data
classes = $CLASSES
train = train.txt
test = test.txt
names = obj.names
backup = ./
EOF

echo "Starting single GPU training"
darknet detector train \
    ./obj.data \
    ./yolo-obj.cfg \
    "$BASEDIR"/pretrained/yolov4.conv.137 \
    -dont_show \
    -gpus 0
