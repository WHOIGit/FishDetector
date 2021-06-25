#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/detect_%j.log

# usage: sbatch detect.sbatch <video> <network_dir>

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


BASEDIR="$(pwd)"

# If there isn't a virtual environment, create it
[ ! -d .venv ] && virtualenv .venv
set +u; . .venv/bin/activate; set -u
pip install -r requirements.txt

# Get configuration options
VIDEOFILE="$1"
NETDIR="$(cd "$2"; pwd)"
NET="${3:-best}"

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# Create a directory to store detections
NETWORKNAME="network_$(basename "$NETDIR")_$NET"
OUTDIR=detections/"$NETWORKNAME"/"$VIDEONAME"

if [ -d "$OUTDIR" ]; then
    echo "Deleting existing directory $OUTDIR"
    rm -Rf "$OUTDIR"
    find detections/byjob -xtype l -delete || true
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
    --progress \
    --ramdisk \
    --num-cores 1 \
    --nn-config "$OUTDIR"/network/yolo-obj.cfg \
    --nn-weights "$OUTDIR"/network/yolo-obj_"$NET".weights \
    --nn-threshold 0.5 \
    --nn-nms 0.4 \
    --save-detection-data "$OUTDIR" \
    $(cat "$OUTDIR"/network/data/settings.txt)
