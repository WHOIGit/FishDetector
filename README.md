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

The reference implementation diverges from the paper by using the [YOLOv3][] object detection algorithm, rather than an R-CNN. We will use [YOLOv4][], training our model using Alexey Bochkovskiy's fork of [Darknet][].

[YOLOv3]: https://pjreddie.com/darknet/yolo/
[YOLOv4]: https://arxiv.org/abs/2004.10934
[Darknet]: https://github.com/AlexeyAB/darknet

[This guide][yolo-guide] explains the distinction between Darknet and YOLO, and [these instructions][instructions] explain in more detail the training process.

[yolo-guide]: https://martinapugliese.github.io/recognise-objects-yolo/
[instructions]: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

1. Prepare the `.txt` files alongside the input files, which we will assume are `.jpg` files stored in `data/`.

2. Clone the [Darknet][] repository and build it with OpenCV support. From here, we will assume that the directory `darknet/` contains the Darknet code, and the `darknet` executable is in the search path.

3. Create the `yolo-obj.cfg` file per the Darknet instructions. A tool is provided in this repository to help:

        $ python configtool.py \
            --classes 1 \
            --batch 64 \
            --subdivisions 8 \
            --no-color-adjustments \
            --size 416 960 \
            darknet/cfg/yolov4-custom.cfg \
            > yolo-obj.cfg

    Also customize the provided `obj.data` and `obj.names` according to instructions.

4. Populate `train.txt` and `test.txt` with paths to input files. Split the input files into the training and testing set randomly. Only include files that have a corresponding bounding box info in a `.txt` file.

    The `generate_train_list.py` script does this:

        python generate_train_list.py --dir data/

5. Download the [pre-trained weights file][weights] (162 MB) to the `pretrained` directory.

[weights]: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

5. Start training:

        darknet detector train obj.data yolo-obj.cfg pretrained/yolov4.conv.137

    You can add `-gpus 0,1,2,...` to utilize multiple GPUs.

    The result of training is a file is called `yolo-obj_best.weights`.

6. To run a test detection, edit `yolo-obj.cfg` and uncomment the `batch` and `subdivisions` settings under the `# Testing` heading, and comment those under the `# Training` heading. Then run:

        darknet detector test obj.data yolo-obj.cfg yolo-obj_final.weights data/20170701145052891_777000.jpg

    To lower the detection threshold, use `-thresh 0.01`.


## Building Darknet with OpenCV on WHOI's HPC

On WHOI's HPC, build Darknet on a GPU node:

    srun -p gpu --pty /bin/bash

Load all necessary modules:

    module load cuda10.1/{toolkit,blas,fft}
    module load cmake/3.14.3 gcc/6.5.0 python3/3.6.5

Create and activate the virtual environment:

    python3 -m virtualenv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    pip uninstall -y opencv-python


### Obtaining cuDNN

WHOI's HPC provides an oudated cuDNN module. A newer version can be obtained from [NVIDIA Developer][cudnn]. Obtain a recent release (tested: 7.6.5) for CUDA 10.1 and unpack it to the `cudnn/` directory (renaming it from `cuda/`).

[cudnn]: https://developer.nvidia.com/cudnn


### Building OpenCV

Darknet requires OpenCV for performing some image manipulation.

Download [OpenCV][opencv-rel] to `opencv/` and [extras][opencv-contrib-rel] to `opencv/opencv_contrib/`.

[opencv-rel]: https://github.com/opencv/opencv/releases
[opencv-contrib-rel]: https://github.com/opencv/opencv_contrib/releases

    mkdir build && cd build

    PYTHON_PREFIX="$(python-config --prefix)"
    PYTHON_LIBRARY="$PYTHON_PREFIX/lib/lib$(python-config --libs | tr ' ' '\n' | cut -c 3- | grep python).so"
    PYTHON_INCLUDE="$(python-config --includes | tr ' ' '\n' | cut -c 3- | head -n 1)"
    PYTHON_PACKAGES="$(python3 -c 'import sys; print(sys.path[-1])')"
    NUMPY_INCLUDE="$(python3 -c 'import numpy; print(numpy.__path__[0])')/core/include"

    mkdir root
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebugInfo \
        -DCMAKE_INSTALL_PREFIX="$(cd root; pwd)" \
        -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
        -DWITH_CUDA=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_CUDNN=ON \
        -DCUDNN_LIBRARY="$(pwd)/../../cudnn/lib64/libcudnn.so" \
        -DCUDNN_INCLUDE_DIR="$(pwd)/../../cudnn/include" \
        -DCUDA_ARCH_BIN=7.0 \
        -DOPENCV_DNN_CUDA=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DPYTHON_DEFAULT_EXECUTABLE="$(command -v python3)" \
        -DPYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE" \
        -DPYTHON3_LIBRARY="$PYTHON_LIBRARY" \
        -DPYTHON3_EXECUTABLE="$(command -v python3)" \
        -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INCLUDE" \
        -DPYTHON3_PACKAGES_PATH="$PYTHON_PACKAGES" \
        -DOPENCV_SKIP_PYTHON_LOADER=ON
    cmake --build . -j "$(nproc)"

Compiling the CUDA source files takes an unusually long time, so the build may appear to stall.

Finally, we can install the Python module to the virtual environment. Be sure to comment `opencv-python` out of the `requirements.txt` file as well, so it is not replaced.

    cp lib/python3/cv2.*.so "$PYTHON_PACKAGES/"


### Building Darknet

    mkdir build_release && cd build_release
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCUDNN_LIBRARY="$(pwd)/../../cudnn/lib64/libcudnn.so" \
        -DCUDNN_INCLUDE_DIR="$(pwd)/../../cudnn/include" \
        -DENABLE_CUDNN_HALF=ON \
        -DOpenCV_DIR=$(cd ../../opencv/build; pwd)
    cmake --build . -j "$(nproc)"


## Building Darknet with OpenCV on macOS

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
