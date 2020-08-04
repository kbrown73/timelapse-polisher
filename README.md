# timelapse-polisher
Temporal and spatial HLS smoothing over image sequence.

Timelapse Polisher analyses the luminance (and optionally hue and saturation) changes of an image sequence both spatially and temporally, smoothes the changes out using Savitzky–Golay filter and then applies corrections to make the changes appear smooth over time. While the script is capable of doing all this pixel by pixel, experiments have shown it is usually best to do the analysing and smoothing on a down scaled and blurred version of the input images. Otherwise you would be correcting for noise and high frequency movements etc.

Here is an example before/after comparison: https://youtu.be/Wwqp80QfCsk

```
usage: timelapse_polisher.py [-h] [-hue] [-sat] [-d DOWNSCALE] [-b BLUR]
                             [-wl WINDOW_LENGTH] [-po POLY_ORDER]
                             [-do DOWNSCALE_OUTPUT] [-pi] [-pl]
                             [-of OUT_FOLDER] [-f] [-sp] [-p]
                             files [files ...]

positional arguments:
  files                 List of files e.g: *.jpg

optional arguments:
  -h, --help            show this help message and exit
  -hue, --hue-smooth    Smooth hues as well. Default = False
  -sat, --saturation-smooth
                        Smooth saturations as well. Default = False
  -d DOWNSCALE, --downscale DOWNSCALE
                        Downscale input image by this factor before measuring
                        luminances. Default = 8.
  -b BLUR, --blur BLUR  Gaussian blur the luminance map by this factor.
                        Default = 20.
  -wl WINDOW_LENGTH, --window-length WINDOW_LENGTH
                        Window length for Savitzky-Golay filter. Default = 51
                        or number of input files whichever is smaller. This
                        will be rounded down to nearest odd number.
  -po POLY_ORDER, --poly-order POLY_ORDER
                        Polynomial order for Savitzky-Golay filter. Default =
                        3. Must be less than window length.
  -do DOWNSCALE_OUTPUT, --downscale-output DOWNSCALE_OUTPUT
                        Downscale the output images by this factor. Default =
                        1.
  -pi, --preview-in     Preview the first input image (half size) and exit.
  -pl, --preview-lum    Preview the first luminance map and exit.
  -of OUT_FOLDER, --out-folder OUT_FOLDER
                        Name of the output folder to be created under current
                        working path. Default = 'df'
  -f, --force           With this flag the output folder will be deleted if it
                        exists and a new one will be created.
  -sp, --show-plot      Show a plot of mean HLS for input and output images
                        with correction graph.
  -p, --preview         Output and show a preview video.
```