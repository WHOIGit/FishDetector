
#!/usr/bin/env python3
#
# Based on "Automatic fish detection in underwater videos by a deep neural 
# network-based hybrid motion learning system" by Salman, et al. (2019). Please
# see the README file for full details.
#
import argparse
import os

import cv2
import numpy
import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True)
    parser.add_argument('-o', '--output', required=True)

    group = parser.add_argument_group('preprocessing')
    group.add_argument('--resize', nargs=2, type=int)

    group = parser.add_argument_group('background subtraction')
    group.add_argument('--bg-gamma', type=float, default=1.5)
    group.add_argument('--bg-history', type=int, default=250)
    group.add_argument('--bg-var-threshold', type=float, default=16.0)

    group = parser.add_argument_group('optical flow')
    group.add_argument('--of-pyr-scale', type=float, default=0.95)
    group.add_argument('--of-levels', type=int, default=10)
    group.add_argument('--of-winsize', type=int, default=15)
    group.add_argument('--of-iterations', type=int, default=3)
    group.add_argument('--of-poly-n', type=int, default=5)
    group.add_argument('--of-poly-sigma', type=float, default=1.2)

    return parser


def main(args):
    video = cv2.VideoCapture(args.video)

    bgsub = cv2.createBackgroundSubtractorMOG2(
        history=args.bg_history,
        varThreshold=args.bg_var_threshold,
        detectShadows=False
    )

    prev = None

    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for nf in tqdm.tqdm(range(nframes)):
        success, frame = video.read()
        assert success

        # Compute the output filename
        name_prefix, _ = os.path.splitext(os.path.basename(args.video))
        timestamp = int(video.get(cv2.CAP_PROP_POS_MSEC))
        out = os.path.join(args.output, '%s_%i.jpg' % (name_prefix, timestamp))

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

        # Visualize the flow magnitude in color
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv = numpy.zeros((frame.shape[0], frame.shape[1], 3))
        hsv[...,0] = numpy.rad2deg(ang) / 2  # hue
        hsv[...,1] = 255  # saturation
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value

        # Apply an opening operator
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)

        # Store the result in the blue channel
        hsvgray = cv2.cvtColor(hsv.astype('float32'), cv2.COLOR_HSV2BGR)
        hsvgray = cv2.cvtColor(hsvgray, cv2.COLOR_BGR2GRAY)
        combined[...,0] = hsvgray


        # Output the combined image
        cv2.imwrite(out, combined)


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
