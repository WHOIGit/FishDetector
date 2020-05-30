import csv
import datetime
import json
import os
import sys

import clickpoints
import cv2

from hopcroftkarp import HopcroftKarp


videos = {
    '20170701145052891': 'data/201707/20170701145052891.avi',
    '20170821170158310': 'data/201708/20170821170158310.avi',
    '20171018152839546': 'data/201710/20171018152839546.avi',
}

cdbs = {
    '20170701145052891':
        'data/clickpoints/20170701145052891_finalstartstopandtrack.cdb',
    '20170821170158310':
        'data/clickpoints/20170821170158310_finalstartstopandtrack.cdb',
    '20171018152839546':
        'data/clickpoints/20171018152839546_MNM_0819.cdb',
}

timestamps = {
    k: datetime.datetime.strptime(
        k + 'Z',
        '%Y%m%d%H%M%S%f%z'
    )
    for k in videos.keys()
}

framerates = {
    k: cv2.VideoCapture(v).get(cv2.CAP_PROP_FPS)
    for k, v in videos.items()
}

frames = {}

# Load all of the tagged points
for video, vidfile in videos.items():
    db = clickpoints.DataFile(cdbs[video])
    for marker in db.getMarkers(type=db.getMarkerType('Fish')):
        k = (video, marker.image.frame)
        if k not in frames:
            frames[k] = {'points': [], 'rects': []}
        frames[k]['points'].append((int(marker.x), int(marker.y)))

# Load all of the files
for p in os.listdir('detections'):
    p = os.path.join('detections', p)
    if not p.endswith('.json'): continue

    with open(p) as f:
        info = json.load(f)

    vid, _ = os.path.splitext(os.path.basename(info['video']))
    k = (vid, info['frame']['number'])
    if k not in frames:
        frames[k] = {'points': [], 'rects': []}
    for det in info['detections']:
        frames[k]['rects'].append((
            det['confidence'],
            (det['left'], det['top'], det['width'], det['height']),
        ))

# Compute timestamps for every frame
for k, info in frames.items():
    vid, fnum = k
    info['timestamp'] = \
        timestamps[vid] + datetime.timedelta(seconds=fnum / framerates[vid])


def contains(rect, point):
    x, y = point
    rx, ry, rw, rh = rect
    return x >= rx and x <= rx + rw and y >= ry and y <= ry + rh

results = {}
for cthreshold in [0.5, 0.75]:
    results[cthreshold] = hourofday = {
        k: [ 0, 0, 0 ] # tp, fp, fn
        for k in range(24)
    }

    for info in frames.values():
        graph = { }
        points = info['points']
        rects = [r for c, r in info['rects'] if c >= cthreshold]
        for pt in points:
            for rect in rects:
                if contains(rect, pt):
                    graph.setdefault(pt, []).append(rect)

        m = HopcroftKarp(graph).maximum_matching(keys_only=True)

        tp = len(m)  # matched points to rects
        fp = len(rects) - len(m)  # rects not matched to points
        fn = len(points) - len(m)  # points not matched to rects

        stats = hourofday[info['timestamp'].hour]
        stats[0] += tp
        stats[1] += fp
        stats[2] += fn

writer = csv.writer(sys.stdout)
for h in range(24):
    row = [h]
    for conf, stats in results.items():
        row.extend(stats[h])
        row.append(None)
        row.append(None)
        row.append(None)
    writer.writerow(row)
print()