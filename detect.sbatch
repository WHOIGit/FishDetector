#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80gb
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/detect_%j.log

set -eu

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date
trap date EXIT

module load python3/3.6.5

BASEDIR="$(pwd)"

# If there isn't a virtual environment, create it
[ ! -d .venv ] && virtualenv .venv
set +u; . .venv/bin/activate; set -u
pip install -r requirements.txt

# Get configuration options
VIDEOFILE="$1"
NETDIR="$(cd "$2"; pwd)"

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# Create a directory to store detections
FINGERPRINT="$(basename "$(readlink "$NETDIR"/data)")"
OUTDIR=detections/"$FINGERPRINT"/"$VIDEONAME"

if [ -d "$OUTDIR" ]; then
    echo "Deleting existing directory $OUTDIR"
    rm -Rf "$OUTDIR"
    find detections/byjob -xtype l -delete
fi

mkdir -p "$OUTDIR"

# Create some links within the output directory
ln -s "$NETDIR" "$OUTDIR"/network
ln -s "$BASEDIR"/logs/detect_"$SLURM_JOB_ID".log "$OUTDIR"/detect.log

# Create a link to find the results by job
mkdir -p detections/byjob || true
ln -s ../../"$OUTDIR" detections/byjob/"$(date +'%Y-%m-%d')"_"$SLURM_JOB_ID"

python process_video.py \
    -v "$VIDEOFILE" \
    --nn-config "$OUTDIR"/network/yolo-obj.cfg \
    --nn-weights "$OUTDIR"/network/yolo-obj_final.weights \
    --nn-threshold 0.5 \
    --save-detection-data "$OUTDIR" \
    $(cat "$OUTDIR"/network/data/settings.txt)
