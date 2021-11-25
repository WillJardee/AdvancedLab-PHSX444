import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

import copy

import astropy.units as u
from astropy.constants import k_B


# ==============================================================================
# Reading in
# ==============================================================================
def read_in(target):
    data = pd.read_csv("group02_data/processed_data/" + target)
    data.rename(columns={'# x': 'x'}, inplace=True)

    data.loc[:, 'x'] = (np.array(data.loc[:, 'x']) * xpix).to(u.mm)
    data.loc[:, 'y'] = (np.array(data.loc[:, 'y']) * ypix).to(u.mm)
    return data


# ==============================================================================
# Analysis
# ==============================================================================
def micromotion_sub(variable, data, dt, region):
    datav = data[variable]
    datav_four = np.fft.fft(datav)
    freq = np.fft.fftfreq(len(datav), dt)
    dffreq = pd.DataFrame({'freq': freq, 'data': datav_four})
    dffreq_sub = copy.deepcopy(dffreq)
    dffreq_sub.loc[(np.abs(dffreq_sub['freq']) > region[0]) & (np.abs(dffreq_sub['freq']) < region[1]), 'data'] = 0

    datav_sub = np.fft.ifft(dffreq_sub['data']).real
    return datav, dffreq, datav_sub, dffreq_sub


def vel_calc(data, dt, bins):
    vel = (data[1:] - data[:-1]) / dt
    vel = np.histogram(vel, bins=bins)
    df = pd.DataFrame(
        {'bin_right': vel[1][1:], 'bin_right_squr': vel[1][1:] ** 2, 'count': vel[0], 'err': np.sqrt(vel[0])})
    return df[df['count'] >= 10]


# ==============================================================================
# Plotting
# ==============================================================================
def plot_pos(data_init, data_sub, freq_init, freq_sub, dt, labels, img_path, color=None):
    if color is None:
        color = color0
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(18)
    spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1], wspace=0.2, hspace=0.5)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])

    t = np.linspace(0, len(data_init) * dt, len(data_init))
    ax0.plot(t, data_init, label=labels['data_label_1'], color=color0[0])
    ax0.plot(t, data_sub, label=labels['data_label_2'], color=color0[1])
    ax0.set_xlabel(labels['data_xlabel'])
    ax0.set_ylabel(labels['data_ylabel'])
    ax0.set_title(labels['data_title'])
    ax0.legend()

    ax1.plot(freq_init['freq'], np.abs(freq_init['data']), color=color0[0], label=labels['freq_label_1'])
    ax1.plot(freq_sub['freq'], np.abs(freq_sub['data']), color=color0[1], label=labels['freq_label_2'])
    ax1.set_xlim(0., np.max(freq_init['freq']))
    ax1.set_xlabel(labels['freq_xlabel'])
    ax1.set_ylabel(labels['freq_ylabel'])
    ax1.set_title(labels['freq_title'])
    ax1.set_yscale('log')
    ax1.legend()

    fig.tight_layout()
    plt.savefig(img_path + labels['save_name'])
    plt.show()


def plot_vel(data, dt, labels, img_path, color=None):
    if color is None:
        color = color0

    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(18)
    spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1], wspace=0.2, hspace=0.5)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])

    fit, cov = np.polyfit(data['bin_right_squr'], np.log(data['count']), 1, w=np.log(data['err']), cov=True)

    x = np.linspace(np.min(data['bin_right_squr']), np.max(data['bin_right_squr']), 1000)
    ax0.errorbar(data['bin_right_squr'], data['count'], yerr=data['err'], linestyle="None", marker="None", color=color[0],
                 label=labels['label_1'])
    ax0.plot(x, np.exp(fit[0] * x + fit[1]), color=color[1])
    ax0.set_xlabel(labels['xlabel_1'])
    ax0.set_ylabel(labels['ylabel_1'])
    ax0.set_title(labels['title_1'])

    plt.errorbar(data['bin_right_squr'], data['count'] - np.exp(fit[0] * data['bin_right_squr'] + fit[1]),
                 yerr=data['err'], linestyle="None", marker="None", color=color[0], label=labels['label_2'])
    plt.hlines(0, np.min(data['bin_right_squr']), np.max(data['bin_right_squr']), color='black')
    ax1.set_xlabel(labels['xlabel_2'])
    ax1.set_ylabel(labels['ylabel_2'])
    ax1.set_title(labels['title_2'])

    fig.tight_layout()
    plt.savefig(img_path + labels['save_name'])
    plt.show()
    return fit, cov[0, 0]


# ==============================================================================
# Calculations
# ==============================================================================

def calc_mass(fit):
    t = 298.15 * u.Kelvin
    m = (fit[0] * (-2 * k_B * t) / (u.mm / u.s) ** 2).to(u.pg)
    return m


def main(sets):
    data = read_in(sets['target'])
    data, freq, data_sub, freq_sub = micromotion_sub(sets['target_variable'], data, sets['time_step'],
                                                     sets['region'])
    plot_pos(data, data_sub, freq, freq_sub, sets['time_step'], sets['pos_labels'], sets['img_path'],
             color=sets['colors'])
    velocities = vel_calc(data_sub, sets['time_step'], sets['bins'])
    fit, chi2 = plot_vel(velocities, sets['time_step'], sets['vel_labels'], sets['img_path'], color=sets['colors'])
    print(f"calculated mass: {calc_mass(fit):.2e}")
    print(f"chi: {chi2:.2e}")


# ==============================================================================
# Settings
# ==============================================================================
if __name__ == "__main__":
    color0 = ['#3387ec', '#ec9833']

    xscale = 66
    yscale = 67
    xpix = u.def_unit('xpix', 1 * u.mm / xscale)
    ypix = u.def_unit('ypix', 1 * u.mm / yscale)

    pos_labels = {
        'data_label_1':     'Initial Data',
        'data_label_2':     r'Inverse FFT (50-80$\rightarrow$0)',
        'data_xlabel':      'Time (s)',
        'data_ylabel':      'Position (mm)',
        'data_title':       'Y Position in Time Domain - 198 Hz',
        'freq_label_1':     'FFT',
        'freq_label_2':     r'FFT (50-80$\rightarrow$0)',
        'freq_xlabel':      'Freq (Hz)',
        'freq_ylabel':      'Amplitude',
        'freq_title':       'Freq Domain - 198 Hz',
        'save_name':        'data_04_y_pos.png',
    }
    vel_labels = {
        'label_1':      '',
        'xlabel_1':     r'Velocity$^2$ (mm/s)$^2$',
        'ylabel_1':     'Counts',
        'title_1':      'Velocity Squared',
        'label_2':      '',
        'xlabel_2':     r'Residuals of Velocity$^2$ (mm/s)$^2$',
        'ylabel_2':     'Counts-Fit',
        'title_2':      'Residuals',
        'save_name':    'data_04_y_vel.png',
    }

    settings = {
        'img_path': './images/',
        'pos_labels': pos_labels,
        'vel_labels': vel_labels,
        'colors': color0,

        'target': 'data_04.csv',
        'time_step': 1 / 198,
        'target_variable': 'y',
        'region': (50, 80),
        'bins': 5000,
    }
    main(settings)
