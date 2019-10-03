#!/usr/bin/env bash

#SBATCH --job-name=FishDetector
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=output_%x.out

echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date
trap date EXIT

module load cuda91/toolkit cuda91/cudnn darknet

# Delete existing weights
rm -f *.weights

# If there is a local build of darknet, prefer it
if [ -e ./darknet ]; then
	PATH="$(pwd)/darknet:$PATH"
fi

print "Starting single GPU training"
darknet detector train obj.data yolo-obj.cfg darknet53.conv.74 -gpus 0 &
TRAIN_PID=$!

# Wait to get to 1000 iterations
while [ 1 ]; do
	if ! kill -0 $TRAIN_PID; then
		exit 1
	fi
	if [ -f yolo-obj_1000.weights ]; then
		echo "First 1000 iterations complete, stopping single GPU training"
		sleep 5
		kill $TRAIN_PID
		wait $TRAIN_PID
		break
	fi
	sleep 10
done

# Start training again, with multiple GPUs this time
print "Starting multi GPU training"
darknet detector train obj.data yolo-obj.cfg yolo-obj_1000.weights -gpus 0,1
