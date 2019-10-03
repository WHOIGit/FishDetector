#!/usr/bin/env python
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--negatives')
parser.add_argument('--train-ratio', default=0.9)
parser.add_argument('--hack', action='store_true')
args = parser.parse_args()


listing = [ os.path.join(args.dir, f) for f in os.listdir(args.dir) ]
files = [ f for f in listing \
        if f.endswith('.jpg') and not f.endswith('_original.jpg') and
           f.replace('.jpg', '.txt') in listing ]


# This is a hack to deal with the fact that we haven't vetted all labels yet
def hack_filter(f):
    prefix, _ = os.path.splitext(os.path.basename(f))
    basefile, _, frameno = prefix.partition('_')
    frameno = int(frameno)

    if basefile == '20170821170158310' and frameno > 6589666:
        return False
    return True


if args.hack:
    files = list(filter(hack_filter, files))


# Add a random set of true negatives
if args.negatives:
    negatives = [ os.path.join(args.negatives, f)
                  for f in os.listdir(args.negatives)
                  if f.endswith('.jpg') and not f.endswith('_original.jpg') ]
    if args.hack:
        negatives = list(filter(hack_filter, negatives))

    # Pick a random set of true negatives to add to the files list
    negatives = random.sample(negatives, len(files))
    files.extend(negatives)

    # Create empty bounding box files
    for f in negatives:
        if not f.replace('.jpg', '.txt') in listing:
            open(f.replace('.jpg', '.txt'), 'w').close()


random.shuffle(files)
ntrain = int(args.train_ratio * len(files))

with open('train.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[:ntrain])

with open('test.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[ntrain:])
