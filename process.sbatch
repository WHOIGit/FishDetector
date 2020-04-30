#!/usr/bin/env bash -eu

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80gb
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/process_%j.log

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date
trap date EXIT

module load python3/3.6.5

# If there isn't a virtual environment, create it
[ ! -d .venv ] && virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Get configuration options
VIDEOFILE=$1
shift
PROCESS_ARGS="$@"

# Get the name of the video without extension
VIDEONAME=$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)

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
OUTDIR=data/"$FINGERPRINT"/"$VID_NAME"
mkdir -p "$OUTDIR"

# Start processing the frames
python process_video.py -v "$VIDEOFILE" $PROCESS_ARGS
