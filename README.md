# timelapse-polisher
Temporal and spatial luminance smoothing over image sequence.

Timelapse Polisher analyses the luminance changes of an image sequence both spatially and temporally, smoothes those changes using Savitzky–Golay filter and then applies corrections to make the changes appear smooth over time. While the script is capable of doing all this pixel by pixel, experiments have shown it is usually best to do the analysing and smoothing on a down scaled and blurred version of the input images. Otherwise you would be correcting for noise and high frequency movements etc.

```
usage: timelapse_polisher.py [-h] [-d DOWNSCALE] [-b BLUR]
                             [-do DOWNSCALE_OUTPUT] [-pi] [-pl]
                             files [files ...]

positional arguments:
  files                 List of files e.g: *.jpg

optional arguments:
  -h, --help            show this help message and exit
  -d DOWNSCALE, --downscale DOWNSCALE
                        Downscale input image by this factor before measuring
                        luminances. Default = 8.
  -b BLUR, --blur BLUR  Gaussian blur the luminance map by this factor.
                        Default = 20.
  -do DOWNSCALE_OUTPUT, --downscale-output DOWNSCALE_OUTPUT
                        Downscale the output images by this factor. Default =
                        1.
  -pi, --preview-in     Preview the first input image (half size) and exit.
  -pl, --preview-lum    Preview the first luminance map and exit.
```