import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

path = './group02_data/stills/'

im = Image.open(path + 'Horzontal calibration bottom is mm.bmp')
plt.imshow(im)
im_data = np.array(im)[1080, :]
plt.plot(im_data, color='black')
plt.show()

freq = np.fft.rfftfreq(im_data.shape[0])
data = np.fft.rfft(im_data)
plt.plot(1/freq, np.abs(data.real))
plt.xlim(0,200)
plt.vlines(66, 0, 25000, color='black')
plt.show()


im = Image.open(path + 'Vertical calibration right is mm.bmp')
plt.imshow(im)
im_data = np.array(im)[:, 900]
plt.plot(im_data, color='black')
plt.show()

freq = np.fft.rfftfreq(im_data.shape[0])
data = np.fft.rfft(im_data)
plt.plot(1/freq, np.abs(data.real))
plt.xlim(0,200)
plt.vlines(67, 0, 25000, color='black')
plt.show()

