#!/usr/bin/env python
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--positives', '-p', action='append')
parser.add_argument('--negatives', '-n', action='append')
parser.add_argument('--train-ratio', default=0.9)
args = parser.parse_args()


# Find positive examples
poslisting = []
for dir in args.positives:
    poslisting.extend(os.path.join(dir, f) for f in os.listdir(dir))
positives = [ f for f in poslisting
              if f.endswith('.jpg') and not f.endswith('_original.jpg') and
                  f.replace('.jpg', '.txt') in poslisting ]


# Find negative examples
neglisting = []
for dir in args.negatives:
    neglisting.extend(os.path.join(dir, f) for f in os.listdir(dir))
negatives = [ f for f in neglisting
              if f.endswith('.jpg') and not f.endswith('_original.jpg') ]


# Pick a random set of true negatives to add to the positives list. Choose the
# same number so that our set is not biased towards true positives.
files = list(positives)
files.extend(random.sample(negatives, len(positives)))

# Create empty bounding box files for the negatives
for f in negatives:
    if f in files and not f.replace('.jpg', '.txt') in neglisting:
        open(f.replace('.jpg', '.txt'), 'w').close()


# Shuffle the set of training files
random.shuffle(files)
ntrain = int(args.train_ratio * len(files))

with open('train.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[:ntrain])

with open('valid.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[ntrain:])
