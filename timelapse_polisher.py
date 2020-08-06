#!/usr/bin/env python3

import os, argparse, shutil, sys
from functools import partial
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import scipy.signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

fim_h = None
fim_l = None
fim_s = None

def read_hls(width, height, downscale, blur, files):
    n = files[0]
    file = files[1]

    print("Reading HLS from: %s" % file)
    im = cv2.imread(file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HLS).astype(float)
    if(downscale > 1):
        im = cv2.resize(im, (width // downscale, height // downscale))

    im /= 255.0
    im_h, im_l, im_s = cv2.split(im)
    im = None

    if(blur > 0.0):
        im_h = gaussian_filter(im_h, sigma = blur)
        im_s = gaussian_filter(im_s, sigma = blur)
        im_l = gaussian_filter(im_l, sigma = blur)

    return im_h, im_l, im_s

def post_process(width, height, downscale_output, out_path, preview, files):
    global fim_h
    global fim_l
    global fim_s

    n = files[0]
    file = files[1]

    im = cv2.imread(file)
    if(downscale_output > 1):
        width = width // downscale_output
        height = height // downscale_output
        im = cv2.resize(im, (width, height))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2HLS).astype(float)

    file = os.path.basename(file)
    tokens = file.split('.')
    tokens[-2] += '_df'
    out_file = '.'.join(tokens)
    out_file = os.path.join(out_path, out_file)

    print("Post processing: %s" % out_file)

    im_h, im_l, im_s = cv2.split(im)

    nfl = fim_l[:, :, n]
    nfl = cv2.resize(nfl, (width, height))
    im_l = np.multiply(im_l, nfl)
    im_l = np.clip(im_l, 0.0, 255.0)
    m_l = im_l.mean() / 255.0

    nfh = None
    m_h = None
    if fim_h is not None:
        nfh = fim_h[:, :, n]
        nfh = cv2.resize(nfh, (width, height))
        im_h = np.multiply(im_h, nfh)
        im_h = np.clip(im_h, 0.0, 255.0)
        m_h = im_h.mean() / 255.0

    nfs = None
    m_s = None
    if fim_s is not None:
        nfs = fim_s[:, :, n]
        nfs = cv2.resize(nfs, (width, height))
        im_s = np.multiply(im_s, nfs)
        im_s = np.clip(im_s, 0.0, 255.0)
        m_s = im_s.mean() / 255.0

    im_output = cv2.merge((im_h, im_l, im_s)).astype(np.uint8)
    im_output = cv2.cvtColor(im_output, cv2.COLOR_HLS2BGR)
    cv2.imwrite(out_file, im_output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if preview:
        factor = 1280.0 / width
        im_output = cv2.resize(im_output, (1280, int(height * factor)))
        return m_h, m_l, m_s, im_output
    else:
        return m_h, m_l, m_s, None

def showplot(im_h, im_l, im_s, ret, out_path):
    global fim_h
    global fim_l
    global fim_s

    im_l_mean = np.zeros(im_l.shape[2])
    im_ls_mean = np.zeros(fim_l.shape[2])
    for i in range(im_l.shape[2]):
        mn = im_l[:, :, i].mean()
        im_l_mean[i] = mn
        mn = fim_l[:, :, i].mean()
        im_ls_mean[i] = mn

    im_h_mean = np.zeros_like(im_l_mean)
    im_hs_mean = np.zeros_like(im_l_mean)
    for i in range(im_h.shape[2]):
        mn = im_h[:, :, i].mean()
        im_h_mean[i] = mn
        if fim_h is not None:
            mn = fim_h[:, :, i].mean()
            im_hs_mean[i] = mn

    im_s_mean = np.zeros_like(im_l_mean)
    im_ss_mean = np.zeros_like(im_l_mean)
    for i in range(im_s.shape[2]):
        mn = im_s[:, :, i].mean()
        im_s_mean[i] = mn
        if fim_s is not None:
            mn = fim_s[:, :, i].mean()
            im_ss_mean[i] = mn

    means = (im_h_mean, im_l_mean, im_s_mean)
    corrections = (im_hs_mean, im_ls_mean, im_ss_mean)
    out_means = (ret[:, 0], ret[:, 1], ret[:, 2])

    plt.rcParams.update({'font.size': 16})
    labels = ['Hue', 'Luminance', 'Saturation']
    figure, axes = plt.subplots(nrows = 2, ncols = 3)
    figure.set_size_inches(20, 15)

    for r, row in enumerate(axes):
        for c, col in enumerate(row):
            if r == 0:
                col.set_title('%s Corrections' % labels[c])
                col.set_ylabel('%s Scalar' % labels[c])
                col.plot(corrections[c], 'g', label = 'Mean %s Correction' % labels[c], linewidth = 2)
            else:
                col.set_title('%s In/Out Means' % labels[c])
                col.set_ylabel('Mean %s' % labels[c])
                col.plot(means[c], 'r', label = 'Input Mean %s' % labels[c], linewidth = 2)
                col.plot(out_means[c], 'b', label = 'Output Mean %s' % labels[c], linewidth = 2)
                col.legend(loc = "upper left")
                col.set_ylim(0.0, 1.0)

    figure.tight_layout(pad = 2.0)
    plt.savefig(os.path.join(out_path, 'stat_graphs.png'))
    plt.show()

def show(im, title = 'image', wait = False):
    cv2.imshow(title, im)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    global fim_h
    global fim_l
    global fim_s

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type = str, nargs = '+', help = "List of files e.g: *.jpg")
    parser.add_argument("-hue", "--hue-smooth", action = 'store_true', help = "Smooth hues as well. Default = False")
    parser.add_argument("-sat", "--saturation-smooth", action = 'store_true', help = "Smooth saturations as well. Default = False")
    parser.add_argument("-d", "--downscale", type = int, help = "Downscale input image by this factor before measuring luminances. Default = 8.")
    parser.add_argument("-b", "--blur", type = float, help = "Gaussian blur the luminance map by this factor. Default = 20.")
    parser.add_argument("-wl", "--window-length", type = int, help = "Window length for Savitzky-Golay filter. Default = 51 or number of input files whichever is smaller. This will be rounded down to nearest odd number.")
    parser.add_argument("-po", "--poly-order", type = int, help = "Polynomial order for Savitzky-Golay filter. Default = 3. Must be less than window length.")
    parser.add_argument("-do", "--downscale-output", type = int, help = "Downscale the output images by this factor. Default = 1.")
    parser.add_argument("-pi", "--preview-in", action = 'store_true', help = "Preview the first input image (half size) and exit.")
    parser.add_argument("-pm", "--preview-hls", action = 'store_true', help = "Preview the first HLS maps and exit.")
    parser.add_argument("-of", "--out-folder", type = str, help = "Name of the output folder to be created under current working path. Default = \'df\'")
    parser.add_argument("-f", "--force", action = 'store_true', help = "With this flag the output folder will be deleted if it exists and a new one will be created.")
    parser.add_argument("-sp", "--show-plot", action = 'store_true', help = "Show a plot of mean HLS for input and output images with correction graph.")
    parser.add_argument("-p", "--preview", action = 'store_true', help = "Output and show a preview video.")
    args = parser.parse_args()

    if len(args.files) < 5:
        raise RuntimeError("Need at least 5 files to process.")

    args.files.sort()
    im = cv2.imread(args.files[0])
    width = im.shape[1]
    height = im.shape[0]

    downscale = 8
    downscale_output = 1
    blur = 20.0
    preview_in = args.preview_in
    preview_hls = args.preview_hls
    wl = min(51, len(args.files))
    po = 3

    if preview_in:
        im = cv2.resize(im, (width // 2, height // 2))
        show(im, wait = True)
        return

    if(args.downscale != None):
        downscale = args.downscale
    if(args.downscale_output != None):
        downscale_output = args.downscale_output
    if(args.blur != None):
        blur = args.blur

    if(args.window_length != None):
        wl = args.window_length
        wl = min(wl, len(args.files))
    if wl % 2 == 0:
        wl -= 1

    if(args.poly_order  != None):
        if(args.poly_order >= wl):
            raise RuntimeError("poly-order must be less than window length.")
        po = args.poly_order

    if preview_hls:
        im_h, im_l, im_s = read_hls(width, height, downscale, blur, (0, args.files[0]))
        show(im_h, title = 'Hue')
        show(im_l, title = 'Luminance')
        show(im_s, title = 'Saturation', wait = True)
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

    cmd = os.path.join(out_path, 'cmd.txt')
    f = open(cmd, 'w')
    f.write(' '.join(sys.argv))
    f.close()

    pool = Pool(cpu_count())
    func = partial(read_hls, width, height, downscale, blur)
    ret = pool.map(func, enumerate(args.files))
    pool.close()
    pool.join()

    ret = np.array(ret)
    ret = np.swapaxes(ret, 0, 1)
    im_h = ret[0].swapaxes(0, 2).swapaxes(0, 1)
    im_l = ret[1].swapaxes(0, 2).swapaxes(0, 1)
    im_s = ret[2].swapaxes(0, 2).swapaxes(0, 1)

    if(args.hue_smooth):
        print("Applying Savitzky-Golay filter (H). Window length = %d, Polyorder = %d" % (wl, po))
        fim_h = scipy.signal.savgol_filter(im_h, wl, po)
        fim_h = np.divide(fim_h, im_h, out = np.zeros_like(fim_h), where = im_h != 0)
        fim_h = np.clip(fim_h, 0.0, 2.0)

    print("Applying Savitzky-Golay filter (L). Window length = %d, Polyorder = %d" % (wl, po))
    fim_l = scipy.signal.savgol_filter(im_l, wl, po)
    fim_l = np.divide(fim_l, im_l, out = np.zeros_like(fim_l), where = im_l != 0)
    fim_l = np.clip(fim_l, 0.0, 2.0)

    if(args.saturation_smooth):
        print("Applying Savitzky-Golay filter (S). Window length = %d, Polyorder = %d" % (wl, po))
        fim_s = scipy.signal.savgol_filter(im_s, wl, po)
        fim_s = np.divide(fim_s, im_s, out = np.zeros_like(fim_s), where = im_s != 0)
        fim_s = np.clip(fim_s, 0.0, 2.0)

    print("Staring post processing...")
    pool = Pool(cpu_count())
    func = partial(post_process, width, height, downscale_output, out_path, args.preview)
    ret = pool.map(func, enumerate(args.files))
    pool.close()
    pool.join()

    ret = np.array(ret)

    if args.preview:
        preview_file = os.path.join(out_path, 'preview.avi')
        print("Writing preview video: %s" % preview_file)
        factor = 1280.0 / width
        preview_out = cv2.VideoWriter(preview_file, cv2.VideoWriter_fourcc('X', '2', '6', '4'), 25, (1280, int(factor * height)))
        frames = ret[:, 3]
        for frame in frames:
            preview_out.write(frame)
        preview_out.release()

    if args.show_plot:
        print("Generating statistics graphs...")
        showplot(im_h, im_l, im_s, ret, out_path)

    if args.preview:
        print("Previewing %s (press \'q\' to quit)" % preview_file)
        run = True
        while run:
            cap = cv2.VideoCapture(preview_file)
            winname = 'Preview (press \'q\' to quit)'
            while True:
                ret, frame = cap.read()

                if ret == True:
                    cv2.imshow(winname, frame)
                else:
                    cap.release()
                    break

                try:
                    if cv2.waitKey(40) & 0xFF == ord('q') or cv2.getWindowProperty(winname, 0) < 0:
                        cap.release()
                        run = False
                        break
                except:
                    cap.release()
                    run = False
                    break

if __name__ == '__main__':
    main()