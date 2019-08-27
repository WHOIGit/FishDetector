This directory contains sample code from the OpenCV project. [License](https://opencv.org/license/).

Additional files are needed:

    wget https://pjreddie.com/media/files/yolov3.weights  
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

The example can be run as follows:

    python object_detection.py \
        --model yolov3.weights \
        --config yolov3.cfg \
        --classes coco_classes.txt \
        --width 416 --height 416 \
        --scale 0.00392 \
        --rgb
