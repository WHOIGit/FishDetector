#!/usr/bin/env python3
import argparse
import collections
import json
import os

import cv2
import intervaltree
import numpy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
    help='path to the video file')
parser.add_argument('-d', '--detections', required=True,
    help='path to a directory of detections')
parser.add_argument('-o', '--output', required=True,
    help='path to a directory to write out video clips')
parser.add_argument('--before', default=3, type=float,
    help='number of seconds\' worth of frames to extract before a fish')
parser.add_argument('--after', default=3, type=float,
    help='number of seconds\' worth of frames to extract after a fish')
parser.add_argument('--every', default=1, type=int,
    help='extract every nth frame')
args = parser.parse_args()


# Load the video
video = cv2.VideoCapture(args.video)
framerate = video.get(cv2.CAP_PROP_FPS)
nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Identify frame numbers with fish
frames_with_fish = set()
for df in os.listdir(args.detections):
    try:
        with open(os.path.join(args.detections, df)) as f:
            info = json.load(f)
            frames_with_fish.add(info['frame']['number'])
    except:
        pass

# Create a tree of intervals before and after each fish frame
# Note: Intervals do not include the upper bound 
tree = intervaltree.IntervalTree()
for frame in frames_with_fish:
    tree[max(0, int(frame - args.before * framerate)):\
         min(int(frame + args.after * framerate) + 1, nframes)] = None

# Simplify the tree to merge overlapping intervals
tree.merge_overlaps()

# Extract the frames
for framenum in tqdm.tqdm(range(tree.end())):
    # Read the timestamp *before* we read the frame
    timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))

    success, frame = video.read()
    if not success:
        continue

    # Check if this frame is in the interval tree
    try:
        interval = next(iter(tree[framenum]))
    except StopIteration:
        continue

    # Output every nth frame from this interval
    if (framenum - interval.begin) % args.every == 0:
        name_prefix, _ = os.path.splitext(os.path.basename(args.video))
        out = '%s_%i' % (name_prefix, timestamp)
        cv2.imwrite(os.path.join(args.output, out + '.jpg'), frame)
