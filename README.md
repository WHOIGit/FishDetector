# Fish Detector

This is an implementation of the fish detection algorithm described by Salman, et al. (2019) [1]. The paper's reference implementation is available [here](https://github.com/ahsan856jalal/Fish-Abundance).


## Datasets

### Fish4Knowledge with Complex Scenes

This dataset [1,2] is comprised of 17 videos from Kavasidis, et al. (2012) and Kavasidis, et al. (2013).

Available [here](http://www.perceivelab.com/index-dataset.php?name=Fish_Detection). Use the "GT - KEY FRAMES" download link.

### Preprocessing Datasets

The videos are provided in the Flash Video (`.flv`) format, which is not widely supported. Use [FFmpeg](https://ffmpeg.org/) to convert files to AVI:

    for x in *.flv; do \
        ffmpeg -i "$x" -c:v mjpeg "$(echo "$x" | sed 's/flv/avi/'); \
    done


## References

1. Salman, A., Siddiqui, S. A., Shafait, F., Mian, A., Shortis, M. R., Khurshid, K., Ulges, A., and Schwanecke, U. *Automatic fish detection in underwater videos by a deep neural network-based hybrid motion learning system.* – ICES Journal of Marine Science, doi:10.1093/icesjms/fsz025.

2. Kavasidis, I., Palazzo, S., Di Salvo, R., Giordano, D., and Spampinato, C., *An innovative web-based collaborative platform for video annotation*, Multimedia Tools and Applications, vol. 70, pp. 413--432, 2013.

3. Kavasidis, I., Palazzo, S., Di Salvo, R, Giordano, D., and Spampinato, C., *A semi-automatic tool for detection and tracking ground truth generation in videos*, Proceedings of the 1st International Workshop on Visual Interfaces for Ground Truth Collection in Computer Vision Applications, pp. 6:1--6:5, 2012.
