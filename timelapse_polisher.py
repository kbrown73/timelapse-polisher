#!/usr/bin/env python3

import os
from PIL import Image
import numpy as np
import scipy.signal
from scipy.ndimage.filters import gaussian_filter
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type = str, nargs = '+', help = "List of files e.g: *.jpg")
    parser.add_argument("-d", "--downscale", type = int, help = "Downscale input image by this factor before measuring luminances. Default = 8.")
    parser.add_argument("-b", "--blur", type = float, help = "Gaussian blur the luminance map by this factor. Default = 20.")
    parser.add_argument("-do", "--downscale-output", type = int, help = "Downscale the output images by this factor. Default = 1.")
    parser.add_argument("-pi", "--preview-in", action = 'store_true', help = "Preview the first input image (half size) and exit.")
    parser.add_argument("-pl", "--preview-lum", action = 'store_true', help = "Preview the first luminance map and exit.")
    args = parser.parse_args()

    if len(args.files) < 5:
        raise RuntimeError("Need at least 5 files to process.")

    args.files.sort()
    im = Image.open(args.files[0])
    width = im.width
    height = im.height

    downscale = 8
    downscale_output = 1
    blur = 20.0
    preview_in = args.preview_in
    preview_lum = args.preview_lum

    if(args.downscale != None):
        downscale = args.downscale
    if(args.downscale_output != None):
        downscale_output = args.downscale_output
    if(args.blur != None):
        blur = args.blur

    nlum = np.zeros((height // downscale, width // downscale, len(args.files)))

    for n, file in enumerate(args.files):
        print("Reading: %s" % file)
        im = Image.open(file)
        if(preview_in):
            im = im.resize((width // 2, height // 2))
            im.show()
            return

        im = im.convert('F')
        print("    Downscaling: %dx" % downscale)
        im = im.resize((width // downscale, height // downscale))


        nim = np.asarray(im) / 255.0
        if(blur > 0.0):
            print("    Blurring: sigma = %f" % blur)
            nim = gaussian_filter(nim, sigma = blur)

        if(preview_lum):
            im = Image.fromarray(nim * 255.0)
            im.show()
            return

        nlum[:, :, n] = nim

    wl = min(51, len(args.files))
    if wl % 2 == 0:
        wl -= 1

    print("Applying Savitzky-Golay filter...")
    nflum = scipy.signal.savgol_filter(nlum, wl, 3)
    nflum = np.divide(nflum, nlum, out = np.zeros_like(nflum), where = nlum!=0)

    out_width = width
    out_height = height
    if(downscale_output > 1):
        out_width = width // downscale_output
        out_height = height // downscale_output

    for n, file in enumerate(args.files):
        print("Post processing: %s" % file)
        im = Image.open(file)
        if(downscale_output > 1):
            print("    Downscaling: %dx" % downscale_output)
            im = im.resize((out_width, out_height))

        tokens = file.split('.')
        tokens[-2] += '_df'
        tokens[-1] = 'jpg'
        out_file = '.'.join(tokens)

        print("    Applying correction...")
        nfl = nflum[:, :, n]
        imfl = Image.fromarray(nfl)
        if(downscale_output > 1):
            imfl = imfl.resize((out_width, out_height))
        else:
            imfl = imfl.resize((width, height))
        nfl = np.asarray(imfl)

        im_r, im_g, im_b = im.split()
        nim_r = np.asarray(im_r)
        nim_g = np.asarray(im_g)
        nim_b = np.asarray(im_b)
        nim_r = np.multiply(nim_r, nfl)
        nim_g = np.multiply(nim_g, nfl)
        nim_b = np.multiply(nim_b, nfl)
        im_r = Image.fromarray(nim_r, 'F').convert('L')
        im_g = Image.fromarray(nim_g, 'F').convert('L')
        im_b = Image.fromarray(nim_b, 'F').convert('L')
        im_output = Image.merge('RGB', (im_r, im_g, im_b))

        print("    Saving: %s" % out_file)
        im_output.save(out_file, quality = 100)

if __name__ == '__main__':
    main()