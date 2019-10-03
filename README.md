# Fish Detector

This is an implementation of the fish detection algorithm described by Salman, et al. (2019) [1]. The paper's reference implementation is available [here](https://github.com/ahsan856jalal/Fish-Abundance).


## Datasets

### Fish4Knowledge with Complex Scenes

This dataset is comprised of 17 videos from Kavasidis, et al. (2012) [2] and Kavasidis, et al. (2013) [3].

Available from [the PeRCeiVe Lab](http://www.perceivelab.com/index-dataset.php?name=Fish_Detection). Use the "GT - KEY FRAMES" download link.

The videos are provided in the Flash Video (`.flv`) format, which is not widely supported. Use [FFmpeg](https://ffmpeg.org/) to convert files to AVI:

    for x in *.flv; do \
        ffmpeg -i "$x" -c:v mjpeg "$(echo "$x" | sed 's/flv/avi/')"; \
    done

### Labeled Fishes in the Wild

This dataset was not used in the paper.

Available from [NOAA](https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/).


## Extracting frames

The `process_video.py` script processes each frame in the input video. A composite image is generated containing the following channels:

  * **Red** - original frame (in grayscale))
  * **Green** - extracted foreground
  * **Blue** - optical flow (mixture of magnitude and angle)


## Labeling objects in frames

The training process expects a `abc.txt` file alongside each input image `abc.jpg` containing a listing of bounding boxes of all objects. For instance,

    0 0.5 0.5 0.10 0.25

represents an object of class 0, centered in the middle of the image, whose width is 10% of the image, and whose height is 25% of the image.

### Labeling software

The [DrawBox](https://github.com/whoigit/DrawBox) tool can be used to help label input images. The output may need to be converted to the above format.

An alternative is [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark).


### Crowdsourced frame labels from Mechanical Turk

[Amazon Mechanical Turk][mturk] can be used to crowdsource (for a small per-image fee) the task of labeling many frames. There is a template for bounding box labeling tasks.

[mturk]: https://www.mturk.com

Labels are returned as straightforward JSON objects, and must be converted to the above format. The `utils/convert_mturk.py` tool provides this functionality.


## Training

The reference implementation diverges from the paper by using the [YOLOv3][] object detection algorithm, rather than an R-CNN. We will also use YOLOv3, training our model using @AlexeyAB's fork of [Darknet][].

[YOLOv3]: https://pjreddie.com/darknet/yolo/
[Darknet]: https://github.com/AlexeyAB/darknet

[This guide][yolo-guide] explains the distinction between Darknet and YOLO, and [these instructions][instructions] explain in more detail the training process.

[yolo-guide]: https://martinapugliese.github.io/recognise-objects-yolo/
[instructions]: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

1. Prepare the `.txt` files alongside the input files, which we will assume are `.jpg` files stored in `data/`.

2. Clone the [Darknet][] repository and build it. From here, we will assume that the directory `darknet/` contains the Darknet code, and the `darknet` executable is in the search path.

    On WHOI's HPC, Darknet is available via the Modules package, however a more recent build of Darknet may be preferable.

        module load cuda91/toolkit cuda91/cudnn darknet

3. Create the `yolo-obj.cfg` file per the Darknet instructions. A patch file is provided in this repository, and can be applied like this:

        patch -o yolo-obj.cfg darknet/cfg/yolov3.cfg < yolo-obj.cfg.patch

    Also customize the provided `obj.data` and `obj.names` according to instructions.

4. Populate `train.txt` and `test.txt` with paths to input files. Split the input files into the training and testing set randomly. Only include files that have a corresponding bounding box info in a `.txt` file.

    The `generate_train_list.py` script does this:

        python generate_train_list.py --dir data/

5. Download the [pre-trained weights file][weights] (154 MB) to the current working directory.

[weights]: https://pjreddie.com/media/files/darknet53.conv.74

5. Start training:

        darknet detector train obj.data yolo-obj.cfg darknet53.conv.74

    You can add `-gpus 0,1,2,...` to utilize multiple GPUs.

    The result of training is a file is called `yolo-obj_best.weights`.

6. To run a test detection, edit `yolo-obj.cfg` and uncomment the `batch` and `subdivisions` settings under the `# Testing` heading, and comment those under the `# Training` heading. Then run:

        darknet detector test obj.data yolo-obj.cfg yolo-obj_final.weights data/20170701145052891_777000.jpg

    To lower the detection threshold, use `-thresh 0.01`.


## Building Darknet on macOS with OpenCV

1. Install OpenCV with `brew install opencv`

2. Configure `pkg-config` to be able to find OpenCV:

        export PKG_CONFIG_PATH=$(brew --prefix opencv)/lib/pkgconfig
        ln -s opencv4.pc $(brew --prefix opencv)/lib/pkgconfig/opencv.pc

3. Modify the `Makefile` to set `OPENCV=1` and `AVX=1`.

4. Run `make`.

5. If you want text labels to appear on the prediction image, copy the `data/labels` directory from the Darknet source directory relative to the path from which you will run the `darknet` command.


## References

1. Salman, A., Siddiqui, S. A., Shafait, F., Mian, A., Shortis, M. R., Khurshid, K., Ulges, A., and Schwanecke, U. *Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system*, ICES Journal of Marine Science, doi:10.1093/icesjms/fsz025, 2019.

2. Kavasidis, I., Palazzo, S., Di Salvo, R., Giordano, D., and Spampinato, C., *An innovative web-based collaborative platform for video annotation*, Multimedia Tools and Applications, vol. 70, pp. 413--432, 2013.

3. Kavasidis, I., Palazzo, S., Di Salvo, R, Giordano, D., and Spampinato, C., *A semi-automatic tool for detection and tracking ground truth generation in videos*, Proceedings of the 1st International Workshop on Visual Interfaces for Ground Truth Collection in Computer Vision Applications, pp. 6:1--6:5, 2012.

4. Cutter, G.; Stierhoff, K.; Zeng, J. *Automated detection of rockfish in unconstrained underwater videos using Haar cascades and a new image dataset: labeled fishes in the wild*, IEEE Winter Conference on Applications of Computer Vision Workshops, pp. 57--62, 2015.
