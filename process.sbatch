#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=logs/process_%j.log

set -eu

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
echo "Command: $0 $@"
hostname; pwd; date
trap date EXIT

[ -n "${CUDA_VISIBLE_DEVICES+x}" ] && HAS_GPU=1 || HAS_GPU=0


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

# Pick a backend for OpenCV DNN
if [ "$HAS_GPU" -eq 1 ]; then
    DNN_BACKEND=CUDA
else
    DNN_BACKEND=OpenCL
fi
echo "[*] Using DNN backend $DNN_BACKEND"

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# Create a fingerprint for the settings used here
SCRIPT_IDENTIFIER="$(git log -n 1 --pretty=format:%H -- process_video.py)"
if git diff --quiet process_video.py; then 
    SCRIPT_IDENTIFIER="$SCRIPT_IDENTIFIER"_dirty
fi
FINGERPRINT="$(echo "SCRIPT_IDENTIFIER";"$PROCESS_ARGS" | openssl md5 | cut -d ' ' -f 2)"

# Create a directory for this fingerprint
if [ ! -d data/"$FINGERPRINT" ]; then
    mkdir -p data/"$FINGERPRINT"
    echo "$SCRIPT_IDENTIFIER" > data/"$FINGERPRINT"/version.txt
    echo "$PROCESS_ARGS" > data/"$FINGERPRINT"/settings.txt
fi

# Create a directory for storing processed frames
OUTDIR=data/"$FINGERPRINT"/"$VIDEONAME"
mkdir -p "$OUTDIR"

# Create a link to find the results by job
mkdir -p data/byjob || true
ln -s ../../"$OUTDIR" data/byjob/"$(date +'%Y-%m-%d')"_"$SLURM_JOB_ID"

# Start processing the frames
python process_video.py \
    -v "$VIDEOFILE" \
    --progress \
    --ramdisk \
    --num-cores 4 \
    --nn-backend "$DNN_BACKEND" \
    $PROCESS_ARGS
