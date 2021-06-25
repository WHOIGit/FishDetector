
#!/usr/bin/env python3
#
# Based on "Automatic fish detection in underwater videos by a deep neural 
# network-based hybrid motion learning system" by Salman, et al. (2019). Please
# see the README file for full details.
#
import argparse
import configparser
import csv
import functools
import json
import multiprocessing
import os
import shutil
import tempfile

import cv2 as cv
import numpy as np
import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--frame-list', action='append')

    group = parser.add_argument_group('acceleration')
    parser.add_argument('-n', '--num-cores', type=int, default=1)
    parser.add_argument('--ramdisk', action='store_true')

    group = parser.add_argument_group('output')
    group.add_argument('--save-original')
    group.add_argument('--save-preprocessed')
    group.add_argument('--save-detection-data')
    group.add_argument('--save-detection-image')

    group = parser.add_argument_group('preprocessing')
    group.add_argument('--resize', nargs=2, type=int)

    group = parser.add_argument_group('background subtraction')
    group.add_argument('--bg-gamma', type=float, default=1.5)
    group.add_argument('--bg-history', type=int, default=250)
    group.add_argument('--bg-var-threshold', type=float, default=16.0)

    group = parser.add_argument_group('optical flow')
    group.add_argument('--of-equalize-luminance', action='store_true')
    group.add_argument('--of-history', action='store_true')
    group.add_argument('--of-use-angle', action='store_true')

    # Values here are from Fish-Abundance, in comments are OpenCV defaults
    group.add_argument('--of-pyr-scale', type=float, default=0.95)  # 0.5
    group.add_argument('--of-levels', type=int, default=10)  # 3
    group.add_argument('--of-winsize', type=int, default=15)  # 7
    group.add_argument('--of-iterations', type=int, default=3)  # 3
    group.add_argument('--of-poly-n', type=int, default=5)  # 5
    group.add_argument('--of-poly-sigma', type=float, default=1.2)  # 1.2

    group = parser.add_argument_group('detection')
    group.add_argument('--nn-threshold', type=float, default=0.5)
    group.add_argument('--nn-nms', type=float, default=0.4)
    group.add_argument('--nn-weights')
    group.add_argument('--nn-config')

    return parser


def num_gpus():
    ordinals = os.environ.get('GPU_DEVICE_ORDINAL', '').split(',')
    return len([o for o in ordinals if o != ''])


def assign_gpu(n):
    assert n < num_gpus()
    cv.cuda.setDevice(n)


def load_network(worker_num, args):
    if not (args.nn_weights and args.nn_config):
        return None, None

    net = cv.dnn.readNet(args.nn_weights, args.nn_config, 'darknet')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Determine the input image size the network expects
    config = configparser.ConfigParser(strict=False)
    config.read(args.nn_config)
    nn_size = (
        int(config['net']['width']),
        int(config['net']['height'])
    )

    return net, nn_size


def main(args):
    # Validate our settings
    outopts = [
        args.save_detection_data,
        args.save_detection_image,
        args.save_original,
        args.save_preprocessed,
    ]
    #assert any(outopts)
    assert all(os.path.isdir(x) for x in outopts if x is not None)

    # If frame lists are provided, create a map from filename_framenumber to
    # the directory to save the output file to.
    if args.frame_list is not None:
        frame_list = {}
        for listfile in args.frame_list:
            with open(listfile) as f:
                for line in f:
                    line = line.rstrip()
                    base, _ = os.path.splitext(os.path.basename(line))
                    if 'negatives/' in line:
                        frame_list[base] = 'negatives/'
                    else:
                        frame_list[base] = './'
        args.frame_list = frame_list

    # Make a copy of the video in RAM for efficiency
    if args.ramdisk:
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')
        copypath = os.path.join(tempdir.name, os.path.basename(args.video))
        shutil.copy(args.video, copypath)
        args.video = copypath

    # Determine the number of frames in the video
    video = cv.VideoCapture(args.video)
    nframes = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    del video

    # Break the frames into work units, scaled by compute power
    power = [1] * args.num_cores
    
    # Total number of frames to process, including duplicates for priming the
    # foreground extraction model.
    wutotframes = nframes + (args.num_cores-1) * args.bg_history

    workunits, start = [], 0
    for p in power:
        wusize = round(p/sum(power) * wutotframes)
        workunits.append((
            # first frame to process
            start,
            # first frame to save
            0 if start == 0 else start + args.bg_history,
            # last frame (excluded)
            start + wusize,
        ))
        start = start + wusize - args.bg_history
    workunits[-1] = (workunits[-1][0], workunits[-1][1], nframes)

    # Kick off the actual thread_main() which processes the video
    if args.num_cores == 1:
        worker_main(args, 0, workunits[0])
    else:
        pool = multiprocessing.Pool(processes=args.num_cores)
        pool.map_async(functools.partial(dispatch_worker, args),
                    enumerate(workunits), chunksize=1)
        pool.close()
        pool.join()


# We can't use a lambda with map_async it seems, so just a dummy dispatch that
# unwraps the (n, workunit) argument
def dispatch_worker(args, nwu):
    worker_main(args, nwu[0], nwu[1])


def worker_main(args, n, workunit):
    # Assign a GPU to us
    assign_gpu(n)

    # Load the neural network
    net, nn_size = load_network(n, args)

    # Open the video file
    video = cv.VideoCapture(args.video)
    
    # For historical reasons, frames are referenced according to their timestamp
    # according to the following function.
    #
    # The frame number should be queried *before* the frame itself is captured.
    #
    # This was tested and matches the number given by CAP_PROP_POS_MSEC. OpenCV
    # uses a frames per second (e.g., 15.0) rather than the microseconds per
    # frame (e.g., 66666) that is actually encoded in the file.
    framerate = video.get(cv.CAP_PROP_FPS)
    get_timestamp = lambda fn: int(1e3 * fn / framerate)

    # Seek to the first frame we care about
    video.set(cv.CAP_PROP_POS_FRAMES, workunit[0])
    assert int(video.get(cv.CAP_PROP_POS_FRAMES)) == workunit[0]

    # Create the optical flow calculator
    flowengine = cv.cuda_FarnebackOpticalFlow.create(
        args.of_levels,
        args.of_pyr_scale,
        False,
        args.of_winsize,
        args.of_iterations,
        args.of_poly_n,
        args.of_poly_sigma,
        0
    )

    # Create the background subtractor
    bgsub = cv.cuda.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=False
    )

    # Create the opening filter
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    filter = cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, cv.CV_8UC4, kernel)

    prev = None

    iterator = range(workunit[0], workunit[2])
    if args.progress:
        iterator = tqdm.tqdm(iterator, position=n, desc=f'G{n:02}')

    # If --of-history is passed, we will use the flow from the previous frame
    # to inform the flow in this frame.
    last_flow = None

    # Compute the prefix of the video name, for building frame filenames later
    name_prefix, _ = os.path.splitext(os.path.basename(args.video))

    stream = cv.cuda_Stream()

    for nf in iterator:
        # Read the next frame
        success, frame = video.read()
        assert success

        # Upload the frame to the device
        frame_local = frame[:]
        frame = cv.cuda_GpuMat(frame.shape[0], frame.shape[1], cv.CV_8UC3)
        frame.upload(frame_local, stream=stream)

        # Resize the frame if necessary
        if args.resize:
            frame = cv.cuda.resize(frame, tuple(args.resize), stream=stream)

        # Compute the output filename
        out = '%s_%i' % (name_prefix, get_timestamp(nf))

        # Determine whether we will want to save this image
        save_this = args.frame_list is None or out in args.frame_list

        # Modify the out path to use whatever directory prefix we are sorting
        # into.
        if args.frame_list is not None:
            out = os.path.join(args.frame_list.get(out, './'), out)


        # -- Foreground extraction --------------------------------------------

        # Apply background subtraction to determine the mask
        mask = bgsub.apply(frame, -1, stream=stream)

        # Store the result in the green channel
        green_channel = mask


        # First exit early spot: If this purely for prepping the background
        # subtractor, we don't need anything further.
        if nf < workunit[1] - 1:
            continue


        # -- Raw image --------------------------------------------------------

        # Convert the frame to grayscale and store it in the red channel
        gray = cv.cuda.cvtColor(frame, cv.COLOR_BGR2GRAY, stream=stream)
        red_channel = gray


        # -- Optical flow -----------------------------------------------------

        # Equalize the luminance histogram of the image
        if args.of_equalize_luminance:
            y, u, v = cv.cuda.split(cv.cuda.cvtColor(frame, cv.COLOR_BGR2YUV, stream=stream), stream=stream)
            y = cv.cuda.equalizeHist(y, stream=stream)

            eqframe = cv.cuda_GpuMat(y.size(), cv.CV_8UC3)
            cv.cuda.merge((y, u, v), eqframe, stream=stream)
            eqframe = cv.cuda.cvtColor(eqframe, cv.COLOR_YUV2RGB,
                                       stream=stream)  # no direct YUV2GRAY
            eqframe = cv.cuda.cvtColor(eqframe, cv.COLOR_RGB2GRAY,
                                       stream=stream)
        else:
            eqframe = gray

        # We can only compute optical flow if there is a previous frame
        preveqframe, prev = prev, eqframe
        if preveqframe is None:
            continue


        # Second early exit spot: We populated preveqframe, we do not need to
        # compute the optical flow for this frame.
        if nf < workunit[1] or not save_this:
            continue

        # Note: We may not want to exit early if args.of_history is on, or we
        # may want to clear history between non-consecutive frames.


        # Compute optical flow between current frame and previous
        flow = flowengine.calc(preveqframe, eqframe, last_flow, stream=stream)
        if args.of_history:
            last_flow = flow

        # Visualize the flow in color
        x, y = cv.cuda.split(flow, stream=stream)
        mag, ang = cv.cuda.cartToPolar(x, y, stream=stream)

        c = cv.cuda_GpuMat(frame.size(), cv.CV_32FC1, 255 / (2*np.pi))
        hue = cv.cuda.multiply(c, ang, stream=stream)
        sat = cv.cuda_GpuMat(frame.size(), cv.CV_32FC1, 255)
        val = cv.cuda.normalize(mag, 0, 255, cv.NORM_MINMAX, -1, stream=stream)

        hsv = cv.cuda_GpuMat(frame.size(), cv.CV_32FC3)
        cv.cuda.merge((hue, sat, val), hsv, stream=stream)

        # Convert to BGRA
        bgr = cv.cuda.cvtColor(hsv, cv.COLOR_HSV2BGR, stream=stream)
        bgra = cv.cuda.cvtColor(bgr, cv.COLOR_BGR2BGRA, stream=stream)

        # Apply an opening operator
        x = cv.cuda_GpuMat(frame.size(), cv.CV_8UC4)
        bgra.convertTo(cv.CV_8UC4, x)
        bgra = filter.apply(x)

        # Store the result in the blue channel
        bgrgray = cv.cuda.cvtColor(bgra, cv.COLOR_BGRA2GRAY, stream=stream)
        blue_channel = bgrgray


        # ---------------------------------------------------------------------

        # Combine the channels
        combined = cv.cuda_GpuMat(blue_channel.size(), cv.CV_8UC3)
        cv.cuda.merge((
            blue_channel,
            green_channel,
            red_channel,
        ), combined, stream=stream)

        # Download the combined image from the device
        output = combined.download(stream=stream)

        # Wait for completion of the stream, after which point the finished
        # image should be in the `output` array.
        stream.waitForCompletion()

        if args.save_original:
            path = os.path.join(args.save_original, out + '_original.jpg')
            cv.imwrite(path, frame_local)
        if args.save_preprocessed:
            path = os.path.join(args.save_preprocessed, out + '.jpg')
            cv.imwrite(path, output)


        # -- Neural network ---------------------------------------------------

        if not net:
            continue

        # Create the blob to feed to the network
        blob = cv.dnn.blobFromImage(
            np.float32(output),
            1/255.0, nn_size, [0, 0, 0], True, crop=False
        )

        # Feed in the blob and get out the detections
        net.setInput(blob)
        nnouts = net.forward(net.getUnconnectedOutLayersNames())

        lastlayer = net.getLayer(net.getLayerId(net.getLayerNames()[-1]))
        assert lastlayer.type == 'Region'

        # Interpret the network output as classification and bounding box
        confidences, boxes = [], []
        for nnout in nnouts:
            for detection in nnout:
                # Determine the classification with the highest confidence
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId].item()
                if confidence < args.nn_threshold:
                    continue
                confidences.append(confidence)

                # Convert the bounding box to absolute coordinates
                height, width, _ = output.shape
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                boxwidth = int(detection[2] * width)
                boxheight = int(detection[3] * height)
                left = int(center_x - boxwidth / 2)
                top = int(center_y - boxheight / 2)
                boxes.append([left, top, boxwidth, boxheight])

        # Apply non-maximum suppression to eliminate overlapping boxes
        indices = \
            cv.dnn.NMSBoxes(boxes, confidences, args.nn_threshold, args.nn_nms)
        indices = indices[:,0] if len(indices) else []

        boxes = [ boxes[i] for i in indices ]
        confidences = [ confidences[i] for i in indices ]

        # Save detection data if desired
        if boxes and args.save_detection_data:
            detout = {
                'video': args.video,
                'frame': {
                    'width': output.shape[1],
                    'height': output.shape[0],
                    'number': nf,
                    'timestamp_msec': get_timestamp(nf),
                }
            }
            detections = detout['detections'] = []

            for conf, box in zip(confidences, boxes):
                detections.append({
                    'left': box[0],
                    'top': box[1],
                    'width': box[2],
                    'height': box[3],
                    'confidence': conf,
                })

            path = os.path.join(args.save_detection_data, out + '_boxes.json')
            with open(path, 'w') as f:
                json.dump(detout, f)

        # Draw labels on an image if desired
        if args.save_detection_image:
            labeled = output.copy()
            for conf, box in zip(confidences, boxes):
                left, top = box[0], box[1]
                right, bot = box[0] + box[2], box[1] + box[3]

                # Draw bounding box
                cv.rectangle(
                    labeled,
                    (left, top),
                    (right, bot),
                    (0, 255, 0)
                )

                # Draw label background
                label = '%.2f' % conf
                labelsz, baseline = \
                    cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(top, labelsz[1])
                cv.rectangle(
                    labeled,
                    (left, top - labelsz[1]),
                    (left + labelsz[0], top + baseline),
                    (255, 255, 255),
                    cv.FILLED
                )

                # Draw label
                cv.putText(labeled, label, (left, top),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Write out file
            path = os.path.join(args.save_detection_image, out + '_labeled.jpg')
            cv.imwrite(path, labeled)


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
