#!/usr/bin/env python3

import os, argparse, shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
import scipy.signal
from scipy.ndimage.filters import gaussian_filter

def read_luma(width, height, downscale, blur, files):
    n = files[0]
    file = files[1]

    print("Reading: %s" % file)
    im = Image.open(file)
    im = im.convert('F')

    if(downscale > 1):
        im = im.resize((width // downscale, height // downscale))

    nim = np.asarray(im) / 255.0
    if(blur > 0.0):
        nim = gaussian_filter(nim, sigma = blur)

    return nim

def post_process(width, height, downscale_output, nflum, out_path, files):
    n = files[0]
    file = files[1]

    im = Image.open(file)
    if(downscale_output > 1):
        width = width // downscale_output
        height = height // downscale_output
        im = im.resize((width, height))

    file = os.path.basename(file)
    tokens = file.split('.')
    tokens[-2] += '_df'
    out_file = '.'.join(tokens)
    out_file = os.path.join(out_path, out_file)

    print("Post processing: %s" % out_file)
    nfl = nflum[:, :, n]
    imfl = Image.fromarray(nfl)
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

    im_output.save(out_file, quality = 100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type = str, nargs = '+', help = "List of files e.g: *.jpg")
    parser.add_argument("-d", "--downscale", type = int, help = "Downscale input image by this factor before measuring luminances. Default = 8.")
    parser.add_argument("-b", "--blur", type = float, help = "Gaussian blur the luminance map by this factor. Default = 20.")
    parser.add_argument("-wl", "--window-length", type = int, help = "Window length for Savitzky-Golay filter. Default = 51 or number of input files whichever is smaller. This will be rounded down to nearest odd number.")
    parser.add_argument("-po", "--poly-order", type = int, help = "Polynomial order for Savitzky-Golay filter. Default = 3. Must be less than window length.")
    parser.add_argument("-do", "--downscale-output", type = int, help = "Downscale the output images by this factor. Default = 1.")
    parser.add_argument("-pi", "--preview-in", action = 'store_true', help = "Preview the first input image (half size) and exit.")
    parser.add_argument("-pl", "--preview-lum", action = 'store_true', help = "Preview the first luminance map and exit.")
    parser.add_argument("-of", "--out-folder", type = str, help = "Name of the output folder to be created under current working path. Default = \'df\'")
    parser.add_argument("-f", "--force", action = 'store_true', help = "With this flag the output folder will be deleted if it exists and a new one will be created.")
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
    wl = min(51, len(args.files))
    po = 3

    if preview_in:
        im = im.resize((width // 2, height // 2))
        im.show()
        return

    if(args.downscale != None):
        downscale = args.downscale
    if(args.downscale_output != None):
        downscale_output = args.downscale_output
    if(args.blur != None):
        blur = args.blur

    if(args.window_length != None):
        wl = args.window_length
    if wl % 2 == 0:
        wl -= 1

    if(args.poly_order  != None):
        if(args.poly_order >= wl):
            raise RuntimeError("poly-order must be less than window length.")
        po = args.poly_order

    if preview_lum:
        ret = read_luma(width, height, downscale, blur, (0, args.files[0]))
        im = Image.fromarray(ret * 255.0)
        im.show()
        return

    folder = 'df'
    if(args.out_folder != None):
        folder = args.out_folder
    out_path = os.path.join(os.getcwd(), folder)

    if os.path.isfile(out_path):
        raise RuntimeError("Cannot create output folder as there is a name conflict with a file of same name.")

    if os.path.isdir(out_path) and not args.force:
        raise RuntimeError("Cannot create output folder as it already exists and -f flag was not used.")

    if os.path.isdir(out_path) and args.force:
        shutil.rmtree(out_path, ignore_errors = True)

    try:
        os.mkdir(out_path)
    except:
        raise RuntimeError("Could not create output path: %s" % out_path)

    pool = Pool(cpu_count())
    func = partial(read_luma, width, height, downscale, blur)
    ret = pool.map(func, enumerate(args.files))
    pool.close()
    pool.join()

    nlum = np.array(ret)
    nlum = np.swapaxes(nlum, 0, 2)
    nlum = np.swapaxes(nlum, 0, 1)

    print("Applying Savitzky-Golay filter. Window length = %d, Polyorder = %d" % (wl, po))
    nflum = scipy.signal.savgol_filter(nlum, wl, po)
    nflum = np.divide(nflum, nlum, out = np.zeros_like(nflum), where = nlum!=0)

    pool = Pool(cpu_count())
    func = partial(post_process, width, height, downscale_output, nflum, out_path)
    pool.map(func, enumerate(args.files))
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()