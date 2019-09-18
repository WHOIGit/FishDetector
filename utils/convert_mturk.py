#!/usr/bin/env python
'''
Converts Mechanical Turk batch result CSV files containing bounding boxes to
training files for Darknet.

Example:

    python convert_mturk.py \
        --approved-only \
        --batch Batch_3761073_batch_results.csv \
        --label Fish 0 \
        --dir directory_of_images/

Files in the directory are matched based on the image URL in the CSV.
'''
import argparse
import csv
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--dir', required=True,
    help='directory of processed frames')
parser.add_argument('-b', '--batch', required=True,
    help='batch results CSV file')
parser.add_argument('--label', nargs=2, action='append', default=[],
    help='case-sensitive label string -> number, e.g., `--label fish 0`')
parser.add_argument('--approved-only', action='store_true',
    help='only export approved tags')
args = parser.parse_args()


args.label = { k: int(v) for k, v in args.label }
print('Assuming label value 0 for all labels without mappings. See --help.')


with open(args.batch, 'r', newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

already_done = set()
for row in rows:
    bn = os.path.basename(row['Input.image_url'])
    approved = row['Approve'] == 'x'
    boxes = json.loads(row['Answer.annotatedResult.boundingBoxes'])
    width = int(row['Answer.annotatedResult.inputImageProperties.width'])
    height = int(row['Answer.annotatedResult.inputImageProperties.height'])

    if not approved and args.approved_only:
        continue

    if bn in already_done:
        print('Warning: Multiple annotations for file %s. Only the last will '
              'be used' % bn)
    already_done.add(bn)

    nameprefix, _ = os.path.splitext(bn)
    with open(os.path.join(args.dir, nameprefix + '.txt'), 'w') as f:
        for box in boxes:
            f.write('%i %.5f %.5f %.5f %.5f\n' % (
                args.label.get(box['label'], 0),
                (box['left'] + (box['width'] / 2)) / width,
                (box['top'] + (box['height'] / 2)) / height,
                box['width'] / width,
                box['height'] / height,
            ))
