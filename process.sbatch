#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=logs/process_%j.log

set -eu

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
echo "Command: $0 $@"
hostname; pwd; date
trap date EXIT

# Error if this is not a GPU node
[ -n "${CUDA_VISIBLE_DEVICES+x}" ]

module purge
module load default-environment slurm/17.11.12
module load gcc/6.5.0 python3/3.6.5

# Load the CUDA modules because OpenCV is linked to them, but we will only use
# them if we're on a GPU node.
module load cuda10.1/{toolkit,blas,fft}
if [ ! -d ./cudnn ]; then
    echo "[*] Missing cuDNN directory, loading cuDNN module"
    module load cuda10.1/cudnn/8.0.2
fi


# If there isn't a virtual environment, create it
[ ! -d .venv ] && virtualenv .venv
set +u; . .venv/bin/activate; set -u
pip install -r requirements.txt

# Get configuration options
VIDEOFILE="$1"
shift
PROCESS_ARGS="$@"

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# Create a fingerprint for the settings used here
SCRIPT_IDENTIFIER="$(git log -n 1 --pretty=format:%H -- process_video.py)"
if git diff --quiet process_video.py; then 
    SCRIPT_IDENTIFIER="$SCRIPT_IDENTIFIER"_dirty
fi
FINGERPRINT="$(echo "$SCRIPT_IDENTIFIER;$PROCESS_ARGS" | openssl md5 | cut -d ' ' -f 2)"

echo "Using fingerprint $FINGERPRINT"

# Create a directory for this fingerprint
if [ ! -d data/"$FINGERPRINT" ]; then
    mkdir -p data/"$FINGERPRINT"
fi

# Save version and settings info
echo "$SCRIPT_IDENTIFIER" > data/"$FINGERPRINT"/version.txt
echo "$PROCESS_ARGS" > data/"$FINGERPRINT"/settings.txt

# Create a directory for storing processed frames
OUTDIR=data/"$FINGERPRINT"/"$VIDEONAME"
mkdir -p "$OUTDIR"

# Create a link to find the results by job
mkdir -p data/byjob || true
ln -s ../../"$OUTDIR" data/byjob/"$(date +'%Y-%m-%d')"_"$SLURM_JOB_ID"

# If frame lists exist, only process those frames
for LIST in test.txt train.txt valid.txt; do
    if [ -f "data/${FINGERPRINT}/${LIST}" ]; then
        echo "Frame list ${LIST} detected, using it"
        PROCESS_ARGS="$PROCESS_ARGS --frame-list data/${FINGERPRINT}/${LIST}"
    fi
done

# Start processing the frames
python process_video.py \
    -v "$VIDEOFILE" \
    --progress \
    --ramdisk \
    --num-cores 2 \
    --save-preprocessed "$OUTDIR" \
    $PROCESS_ARGS
