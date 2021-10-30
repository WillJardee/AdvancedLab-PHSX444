#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze image displacement with cross-correlation
2021-10-05 created by Brian R. D'Urso
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from skimage.registration import phase_cross_correlation as register_translation
# from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

# ==============================================================================
# Settings
# ==============================================================================

N_horizontal_pixels = 50
N_vertical_pixels = 50
image_max_value = 255

directory_images = "./trap23_prossessed"
directory_results = "./"
output_file = "data_23xy.csv"

N_images = 5000


upsample_factor = 100


# ==============================================================================
# Functions
# ==============================================================================

class State(object):
    pass


def setup(state, image_scale_max, COM_marker_radius=0, COM_marker_color=None):
    state.fig = plt.figure()
    plt.gray()  # show grayscale

    # plot for images
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.set_title("Image")

    init_array = np.zeros(shape=(N_vertical_pixels, N_horizontal_pixels), dtype=np.uint16)
    init_array[0, 0] = image_scale_max  # this value allow imshow to initialise it's color scale
    state.zero_array = init_array

    state.imshow = state.ax.imshow(init_array)

    if COM_marker_color is not None:
        state.circle = Circle((0, 0), COM_marker_radius, color=COM_marker_color)
        state.ax.add_patch(state.circle)


def load_image(directory, n):
    # filename = os.path.join(directory, "image_{:05d}.png".format(n))
    im = Image.open(f"{directory}/image_{n:05d}.png")
    image_array = np.array(im.getdata()).reshape(im.size[1], im.size[0])

    return image_array


def analyze_CC(state, image, last_eigenframe, next_eigenframe, plot=True):
    image_fft = np.fft.fft2(image)
    if last_eigenframe is None:
        # this is the first cross-correlation, so use the first image FFT as the eigenframe
        last_eigenframe = image_fft.copy()
        next_eigenframe = image_fft.copy()
        shift = np.array([0.0, 0.0])
        error = 0.0
    else:
        # cross-correlate the images
        shift, error, phasediff = register_translation(last_eigenframe, image_fft, upsample_factor=upsample_factor,
                                                       space="fourier")
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
        state.circle.center = N_horizontal_pixels / 2.0 - shift[1], N_vertical_pixels / 2.0 - shift[0]
        # push update
        state.fig.canvas.flush_events()

    return last_eigenframe, next_eigenframe, (shift[1], shift[0], error)


# ==============================================================================
# Actions
# ==============================================================================

plt.ion()

state = State()
setup(state, image_scale_max=image_max_value, COM_marker_color='red')

eigenframe_stage_1 = None
eigenframe_stage_2 = None
eigenframe_stage_3 = None
eigenframe_stage_4 = None

# array to store the results
data_CC = np.zeros((N_images, 3))

print("Processing data images stage 1...")

for n in range(N_images):
    image = load_image(directory_images, n)
    eigenframe_stage_1, eigenframe_stage_2, data_CC[n] = analyze_CC(state, image, eigenframe_stage_1,
                                                                    eigenframe_stage_2, plot=True)
    print("Data image set {:05d} of {} processed.".format(n + 1, N_images))

eigenframe_stage_2 = eigenframe_stage_2 / N_images

print("Processing data images stage 2...")

for n in range(N_images):
    image = load_image(directory_images, n)
    eigenframe_stage_2, eigenframe_stage_3, data_CC[n] = analyze_CC(state, image, eigenframe_stage_2,
                                                                    eigenframe_stage_3, plot=True)
    print("Data image set {:05d} of {} processed.".format(n + 1, N_images))

eigenframe_stage_3 = eigenframe_stage_3 / N_images

print("Processing data images stage 3...")

for n in range(N_images):
    image = load_image(directory_images, n)
    eigenframe_stage_3, eigenframe_stage_4, data_CC[n] = analyze_CC(state, image, eigenframe_stage_3,
                                                                    eigenframe_stage_4, plot=True)
    print("Data image set {:05d} of {} processed.".format(n + 1, N_images))

eigenframe_stage_4 = eigenframe_stage_4 / N_images

# save the results
filename = os.path.join(directory_results, output_file)
np.savetxt(filename, data_CC, delimiter=',', header="x,y,error")

plt.ioff()
plt.close()
