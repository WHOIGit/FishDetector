
#!/usr/bin/env python3
#
# Based on "Automatic fish detection in underwater videos by a deep neural 
# network-based hybrid motion learning system" by Salman, et al. (2019). Please
# see the README file for full details.
#
import argparse
import configparser
import csv
import json
import os

import cv2
import numpy
import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True)

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
    group.add_argument('--nn-backend', default='OpenCL')
    group.add_argument('--nn-threshold', type=float, default=0.5)
    group.add_argument('--nn-weights')
    group.add_argument('--nn-config')

    return parser


def main(args):
    video = cv2.VideoCapture(args.video)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # This feature has been removed for now
    whitelist = blacklist = None
    if False:
        if args.only_in or args.except_in:
            whitelist = set()
            with open(args.only_in or args.except_in, newline='') as f:
                for row in csv.DictReader(f):
                    whitelist.add(os.path.basename(row['image_url']))

        if args.except_in:
            blacklist, whitelist = whitelist, None

    bgsub = cv2.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=False
    )

    prev = None

    # Load the neural network
    net = None
    if args.nn_weights and args.nn_config:
        net = cv2.dnn.readNet(args.nn_weights, args.nn_config, 'darknet')
        if args.nn_backend == 'OpenCL':
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        elif args.nn_backend == 'CUDA':
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise Exception('Unknown DNN backend ' + args.nn_backend)

        # Determine the input image size the network expects
        config = configparser.ConfigParser(strict=False)
        config.read(args.nn_config)
        nn_size = (
            int(config['net']['width']),
            int(config['net']['height'])
        )

        assert args.save_detection_data or args.save_detection_image

    for nf in tqdm.tqdm(range(nframes)):
        # Read the timestamp *before* we read the frame
        timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))

        # Read the frame itself
        success, frame = video.read()
        assert success

        # Compute the output filename
        name_prefix, _ = os.path.splitext(os.path.basename(args.video))
        out = '%s_%i' % (name_prefix, timestamp)

        # Resize the frame if necessary
        if args.resize:
            frame = cv2.resize(frame, tuple(args.resize))

        # We will put the combined frame here
        combined = numpy.zeros((frame.shape[0], frame.shape[1], 3))


        # -- Raw image --------------------------------------------------------

        # Convert the frame to grayscale and store it in the red channel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        combined[...,2] = gray


        # -- Foreground extraction --------------------------------------------

        # Adjust the gamma of the image
        gamma_adj = ((gray / 255) ** args.bg_gamma) * 255

        # Apply background subtraction to determine the mask
        mask = bgsub.apply(frame)

        # Store the result in the green channel
        combined[...,1] = mask


        # -- Optical flow -----------------------------------------------------

        # Equalize the luminance histogram of the image before converting to
        # grayscale.
        eqframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        eqframe[:,:,0] = cv2.equalizeHist(eqframe[:,:,0])
        eqframe = cv2.cvtColor(eqframe, cv2.COLOR_YUV2RGB)  # no direct 2GRAY
        eqframe = cv2.cvtColor(eqframe, cv2.COLOR_RGB2GRAY)

        # We can only compute optical flow if there is a previous frame
        preveqframe, prev = prev, eqframe
        if preveqframe is None:
            continue

        # If we wanted to skip doing any further processing of this frame,
        # now would be a good time to do it...
        if whitelist and os.path.basename(out) not in whitelist:
            continue
        if blacklist and os.path.basename(out) in blacklist:
            continue

        # Compute optical flow between current frame and previous
        flow = cv2.calcOpticalFlowFarneback(
            preveqframe, eqframe, None,
            args.of_pyr_scale,
            args.of_levels,
            args.of_winsize,
            args.of_iterations,
            args.of_poly_n,
            args.of_poly_sigma,
            0
        )

        # Visualize the flow in color
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv = numpy.zeros((frame.shape[0], frame.shape[1], 3), numpy.uint8)
        hsv[...,0] = 255 * ang / (2*numpy.pi)  # hue
        hsv[...,1] = 255  # saturation
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value

        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply an opening operator
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bgr = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, kernel)

        # Store the result in the blue channel
        bgrgray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        combined[...,0] = bgrgray


        # Output the combined image
        if args.save_preprocessed:
            path = os.path.join(args.save_preprocessed, out + '.jpg')
            cv2.imwrite(path, combined)
        if args.save_original:
            path = os.path.join(args.save_original, out + '_original.jpg')
            cv2.imwrite(path, frame)


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
