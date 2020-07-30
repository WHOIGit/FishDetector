
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

import cv2
import numpy
import tqdm

import cva


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-cores', type=int, default=1)
    parser.add_argument('-v', '--video', required=True)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--ramdisk', action='store_true')
    parser.add_argument('--gpu-factor', default=14.2)  # empirical

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

    # Values here are from Fish-Abundance, in comments are OpenCV defaults
    group = parser.add_argument_group('optical flow')
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


def num_gpus(args):
    return len(os.environ.get('GPU_DEVICE_ORDINAL', '').split(','))


def assign_gpu(worker_num, args):
    if args.nn_backend == 'CUDA':
        if worker_num < num_gpus(args):
            cv2.cuda.setDevice(worker_num)
            return worker_num
    return None


def load_network(worker_num, is_gpu, args):
    if not (args.nn_weights and args.nn_config):
        return None, None

    net = cv2.dnn.readNet(args.nn_weights, args.nn_config, 'darknet')
    if is_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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
    assert any(outopts)
    assert all(os.path.isdir(x) for x in outopts if x is not None)

    # Make a copy of the video in RAM for efficiency
    if args.ramdisk:
        tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')
        copypath = os.path.join(tempdir.name, os.path.basename(args.video))
        shutil.copy(args.video, copypath)
        args.video = copypath

    # Determine the number of frames in the video
    video = cv2.VideoCapture(args.video)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    del video

    # Break the frames into work units, scaled by compute power
    power = [1] * args.num_cores
    for i in range(min(num_gpus(args), args.num_cores)):
        power[i] = args.gpu_factor
    
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
    # Get the OpenCV accelerator if available
    ctx = cva.get_accelerator(n)
    ctx.push_stream()  # reuse across all frames

    # Load the neural network
    net, nn_size = load_network(n, ctx.is_gpu, args)

    # Open the video file
    video = cv2.VideoCapture(args.video)
    
    # Figure out how to measure time
    assert int(video.get(cv2.CAP_PROP_POS_FRAMES)) == 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    assert int(video.get(cv2.CAP_PROP_POS_FRAMES)) == 1

    if int(video.get(cv2.CAP_PROP_POS_MSEC)) == 0:
        framerate = video.get(cv2.CAP_PROP_FPS)
        get_timestamp = lambda nf: int(1000 * (nf / framerate))
    else:
        get_timestamp = lambda _: int(video.get(cv2.CAP_PROP_POS_MSEC))

    # Seek to the first frame we care about
    video.set(cv2.CAP_PROP_POS_FRAMES, workunit[0])
    assert int(video.get(cv2.CAP_PROP_POS_FRAMES)) == workunit[0]

    # Create the optical flow calculator
    flowengine = ctx.FarnebackOpticalFlow.create(
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
    bgsub = ctx.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=False
    )

    # Create the opening filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # FIXME: The image type we pass to the filter is 8UC3, but filter doesn't
    # support it
    #filter = ctx.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC4, kernel)

    prev = None

    iterator = range(workunit[0], workunit[2])
    if args.progress:
        iterator = tqdm.tqdm(iterator, position=n,
                             desc=f'{"G" if ctx.is_gpu else "C"}{n:02}')

    for nf in iterator:
        # Determine the timestamp *before* we read the frame
        timestamp = get_timestamp(nf)

        # Read the frame itself
        success, frame = video.read()
        assert success

        # Compute the output filename
        name_prefix, _ = os.path.splitext(os.path.basename(args.video))
        out = '%s_%i' % (name_prefix, timestamp)

        # Upload the frame to the device
        frame_local = frame[:]
        frame = ctx.upload(frame)

        # Resize the frame if necessary
        if args.resize:
            frame = ctx.resize(frame, tuple(args.resize))


        # -- Raw image --------------------------------------------------------

        # Convert the frame to grayscale and store it in the red channel
        gray = ctx.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        red_channel = gray


        # -- Foreground extraction --------------------------------------------

        # Apply background subtraction to determine the mask
        #mask = bgsub.apply(frame, None, -1, None)
        mask = gray

        # Store the result in the green channel
        green_channel = mask


        # -- Optical flow -----------------------------------------------------

        # Equalize the luminance histogram of the image before converting to
        # grayscale.
        y, u, v = ctx.split(ctx.cvtColor(frame, cv2.COLOR_BGR2YUV))
        y = ctx.equalizeHist(y)

        eqframe = ctx.merge((y, u, v))
        eqframe = ctx.cvtColor(eqframe, cv2.COLOR_YUV2RGB)  # no direct 2GRAY
        eqframe = ctx.cvtColor(eqframe, cv2.COLOR_RGB2GRAY)

        # Exit early if we are just seeding the background subtraction history
        # and not actually processing this frame.
        if nf < workunit[1]:
            continue

        # We can only compute optical flow if there is a previous frame
        preveqframe, prev = prev, eqframe
        if preveqframe is None:
            continue

        # Compute optical flow between current frame and previous
        flow = flowengine.calc(preveqframe, eqframe, None)

        # Visualize the flow in color
        x, y = ctx.split(flow)
        mag, ang = ctx.cartToPolar(x, y)

        # hsv = numpy.zeros((frame.shape[0], frame.shape[1], 3), numpy.uint8)
        # hsv[...,0] = 255 * ang / (2*numpy.pi)  # hue
        # hsv[...,1] = 255  # saturation
        # hsv[...,2] = ctx.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value

        hsv = ctx.merge((mag, mag, mag))

        # Convert to BGR
        bgr = ctx.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply an opening operator
        #bgr = filter.apply(bgr)

        # Store the result in the blue channel
        bgrgray = ctx.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blue_channel = red_channel #bgrgray

        # ---------------------------------------------------------------------

        # Combine the channels
        combined = ctx.merge((
            blue_channel,
            green_channel,
            red_channel,
        ))

        # Download the combined image from the device
        combined = ctx.download(combined)
        ctx.await_stream()

        # Output the combined image
        if args.save_preprocessed:
            path = os.path.join(args.save_preprocessed, out + '.jpg')
            cv2.imwrite(path, combined)
        if args.save_original:
            ctx.await_stream()
            path = os.path.join(args.save_original, out + '_original.jpg')
            cv2.imwrite(path, frame_local)


        # -- Neural network ---------------------------------------------------

        if not net:
            continue

        # Create the blob to feed to the network
        blob = cv2.dnn.blobFromImage(
            numpy.float32(combined),
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
                classId = numpy.argmax(scores)
                confidence = scores[classId].item()
                if confidence < args.nn_threshold:
                    continue
                confidences.append(confidence)

                # Convert the bounding box to absolute coordinates
                height, width, _ = combined.shape
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                boxwidth = int(detection[2] * width)
                boxheight = int(detection[3] * height)
                left = int(center_x - boxwidth / 2)
                top = int(center_y - boxheight / 2)
                boxes.append([left, top, boxwidth, boxheight])

        # Apply non-maximum suppression to eliminate overlapping boxes
        indices = \
            cv2.dnn.NMSBoxes(boxes, confidences, args.nn_threshold, args.nn_nms)
        indices = indices[:,0] if len(indices) else []

        boxes = [ boxes[i] for i in indices ]
        confidences = [ confidences[i] for i in indices ]

        # Save detection data if desired
        if boxes and args.save_detection_data:
            output = {
                'video': args.video,
                'frame': {
                    'width': combined.shape[1],
                    'height': combined.shape[0],
                    'number': nf,
                    'timestamp_msec': timestamp,
                }
            }
            detections = output['detections'] = []

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
                json.dump(output, f)

        # Draw labels on an image if desired
        if args.save_detection_image:
            labeled = combined.copy()
            for conf, box in zip(confidences, boxes):
                left, top = box[0], box[1]
                right, bot = box[0] + box[2], box[1] + box[3]

                # Draw bounding box
                cv2.rectangle(
                    labeled,
                    (left, top),
                    (right, bot),
                    (0, 255, 0)
                )

                # Draw label background
                label = '%.2f' % conf
                labelsz, baseline = \
                    cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(top, labelsz[1])
                cv2.rectangle(
                    labeled,
                    (left, top - labelsz[1]),
                    (left + labelsz[0], top + baseline),
                    (255, 255, 255),
                    cv2.FILLED
                )

                # Draw label
                cv2.putText(labeled, label, (left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Write out file
            path = os.path.join(args.save_detection_image, out + '_labeled.jpg')
            cv2.imwrite(path, labeled)


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
