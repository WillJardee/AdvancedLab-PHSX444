#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze image displacement with cross-correlation
2021-10-05 created by Brian R. D'Urso
2021-11-24 modified by William V. Jardee
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from skimage.registration import phase_cross_correlation as register_translation
# from skimage.feature import register_translation
from scipy.ndimage import fourier_shift


# ==============================================================================
# Progress bar
# ==============================================================================

def processing_flush(n, index=50):
    # prints a progress bar that is 50 dots long.
    sys.stdout.write("\rProcessing %s%s" % ((n % index + 1) * ".", (index - (n % index)) * " "))
    sys.stdout.write("\rProcessing %s" % ((n % index + 1) * "."))
    sys.stdout.flush()


# ==============================================================================
# Functions
# ==============================================================================

class State(object):
    pass


def setup(state, image_scale_max, com_marker_radius=0, com_marker_color=None):
    state.fig = plt.figure()
    plt.gray()  # show grayscale

    # plot for images
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.set_title("Image")

    init_array = np.zeros(shape=(settings['N_vertical_pixels'], settings['N_horizontal_pixels']), dtype=np.uint16)
    init_array[0, 0] = image_scale_max  # this value allow imshow to initialise it's color scale
    state.zero_array = init_array

    state.imshow = state.ax.imshow(init_array)

    if com_marker_color is not None:
        state.circle = Circle((0, 0), com_marker_radius, color=com_marker_color)
        state.ax.add_patch(state.circle)


def load_image(directory, n):
    # filename = os.path.join(directory, "image_{:05d}.png".format(n))
    im = Image.open(f"{directory}/image_{n:05d}.png")
    image_array = np.array(im.getdata()).reshape(im.size[1], im.size[0])

    return image_array


def analyze_cc(state, image, last_eigenframe, next_eigenframe, plot=True):
    image_fft = np.fft.fft2(image)
    if last_eigenframe is None:
        # this is the first cross-correlation, so use the first image FFT as the eigenframe
        last_eigenframe = image_fft.copy()
        next_eigenframe = image_fft.copy()
        shift = np.array([0.0, 0.0])
        error = 0.0
    else:
        # cross-correlate the images
        shift, error, phasediff = register_translation(last_eigenframe, image_fft,
                                                       upsample_factor=settings['upsample_factor'], space="fourier")
        # build the next eigenframe
        # shift the Fourier space image to overlap with the others
        image_fft_translated = fourier_shift(image_fft, shift)
        if next_eigenframe is None:
            next_eigenframe = image_fft_translated.copy()
        else:
            next_eigenframe += image_fft_translated
    if plot:
        # show image
        state.imshow.set_data(image)
        # update marker to indicate shift
        state.circle.center = settings['N_horizontal_pixels'] / 2.0 - shift[1], \
                              settings['N_vertical_pixels'] / 2.0 - shift[0]
        # push update
        state.fig.canvas.flush_events()

    return last_eigenframe, next_eigenframe, (shift[1], shift[0], error)


# ==============================================================================
# Actions
# ==============================================================================

def run_stage(stage_n0, run_num, data_cc, state):
    print(f"Processing data images stage {run_num} of {settings['num_runs']}:")
    stage_n1 = None
    for n in range(settings['N_images']):
        image = load_image(settings['directory_images'], n)
        stage_n0, stage_n1, data_cc[n] = analyze_cc(state, image, stage_n0, stage_n1, plot=True)
        processing_flush(int(settings['progress_bar_len'] * n / settings['N_images']), settings['progress_bar_len'])
    print()
    stage_n1 = stage_n1 / settings['N_images']
    return stage_n1, data_cc


def main(sets):
    plt.ion()

    state = State()
    setup(state, image_scale_max=sets['image_max_value'], com_marker_color='red')

    # array to store the results
    data_cc = np.zeros((sets['N_images'], 3))

    print('Running analysis for ' + sets["directory_images"])
    print('Target: ' + sets['directory_results'] + "/" + sets['output_file'])
    print(" % % %" + " " * 4 + "|" + "=" * (sets['progress_bar_len']) + "|" + "\n")

    run_lis = [None] * (sets['num_runs'] + 1)
    for run in range(sets['num_runs']):
        run_lis[run + 1], data_cc = run_stage(run_lis[run], run + 1, data_cc, state)

    # save the results
    filename = os.path.join(sets['directory_results'], sets['output_file'])
    np.savetxt(filename, data_cc, delimiter=',', header="x,y,error")

    print('Processing done. File can be found at: ' + filename.replace("\\", "/"))
    plt.ioff()
    plt.close()


# ==============================================================================
# Settings
# ==============================================================================
if __name__ == "__main__":
    settings = {
        'progress_bar_len': 50,

        'N_horizontal_pixels': 220,
        'N_vertical_pixels': 220,
        'image_max_value': 255,

        'directory_images': "./group02_data/198 Hz fist bump #1",
        'directory_results': "./group02_data/processed_data",
        'output_file': "data_05.csv",

        'N_images': 2000,
        'upsample_factor': 100,
        'num_runs': 4,
    }
    main(settings)
