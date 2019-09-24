#!/usr/bin/env python
import argparse
import random
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--train-ratio', default=0.9)
args = parser.parse_args()


listing = [ os.path.join(args.dir, f) for f in os.listdir(args.dir) ]
files = [ f for f in listing \
        if f.endswith('.jpg') and f.replace('.jpg', '.txt') in listing ]

random.shuffle(files)
ntrain = int(args.train_ratio * len(files))

with open('train.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[:ntrain])

with open('test.txt', 'w') as o:
    o.writelines(f + '\n' for f in files[ntrain:])
