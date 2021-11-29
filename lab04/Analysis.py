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
    # data = pd.read_csv("group02_data/processed_data/" + target)
    data = pd.read_csv("processed_data/" + target)
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
    vel = np.histogram(np.abs(vel), bins=bins)
    df = pd.DataFrame(
        {'bin_right': vel[1][1:], 'bin_right_squr': vel[1][1:] ** 2, 'count': vel[0], 'err': np.sqrt(vel[0])})
    return df[df['count'] >= 10]


def calc_mass(fit):
    t = 298.15 * u.Kelvin
    m = (fit * (-2 * k_B * t) / (u.mm / u.s) ** 2).to(u.pg)
    return m


# ==============================================================================
# Plotting
# ==============================================================================
def plot_pos(data_init, data_sub, freq_init, freq_sub, dt, labels, img_path, color=None):
    if color is None:
        color = color0

    # Plot setup
    X = [(1, 3, (2, 3)), (1, 3, 1)]
    plt.figure(figsize=(18, 3))
    # plt.subplots_adjust(bottom=0, left=0, top=0.975, right=1)

    # Plotting Time Domain
    plt.subplot(*X[0])
    t = np.linspace(0, len(data_init) * dt, len(data_init))
    plt.plot(t, data_init, label=labels['data_label_1'], color=color0[0])
    plt.plot(t, data_sub, label=labels['data_label_2'], color=color0[1])
    plt.xlim(t[0], t[-1])
    plt.xlabel(labels['data_xlabel'], fontsize=labels['label_fontsize'])
    plt.ylabel(labels['data_ylabel'], fontsize=labels['label_fontsize'])
    plt.title(labels['data_title'], fontsize=labels['title_fontsize'])
    plt.legend(fontsize=labels['legend_fontsize'])

    # Plotting Freq Domain
    plt.subplot(*X[1])
    plt.plot(freq_init['freq'], np.abs(freq_init['data']), color=color0[0], label=labels['freq_label_1'])
    plt.plot(freq_sub['freq'], np.abs(freq_sub['data']), color=color0[1], label=labels['freq_label_2'])
    plt.xlim(0., np.max(freq_init['freq']))
    plt.xlabel(labels['freq_xlabel'], fontsize=labels['label_fontsize'])
    plt.ylabel(labels['freq_ylabel'], fontsize=labels['label_fontsize'])
    plt.title(labels['freq_title'], fontsize=labels['title_fontsize'])
    plt.yscale('log')
    plt.legend(fontsize=labels['legend_fontsize'])

    # Managing final presentation
    plt.tight_layout()
    plt.savefig(img_path + labels['save_name'])
    plt.show()


def plot_vel(data, dt, labels, img_path, color=None):
    if color is None:
        color = color0

    # Plot setup
    X = [(1, 3, (1, 2)), (1, 3, 3)]
    plt.figure(figsize=(18, 3))
    plt.subplots_adjust(bottom=0, left=0, top=0.975, right=1)

    # Calculating fit
    fit, res, _, _, _ = np.polyfit(np.array(data['bin_right_squr']), np.array(np.log(data['count'])), 1,
                                    w =np.log(data['err']), cov=True, full=True)
    fit, cov = np.polyfit(data['bin_right_squr'], np.log(data['count']), 1, w=np.log(data['err']), cov=True)
    chisq_red = float(res / (len(data['bin_right_squr']) - 2))
    err_a = np.sqrt(cov[0, 0])

    # Plotting Velocities and fit
    plt.subplot(*X[0])
    x = np.linspace(np.min(data['bin_right_squr']), np.max(data['bin_right_squr']), 1000)
    plt.errorbar(data['bin_right_squr'], data['count'], yerr=data['err'],
                 linestyle="None", marker="None", ms=7, ecolor=color[0], mfc=color[1], mew=0, label=labels['label_1'])
    plt.plot(x, np.exp(fit[0] * x + fit[1]), color='black')
    plt.xlabel(labels['xlabel_1'], fontsize=labels['label_fontsize'])
    plt.ylabel(labels['ylabel_1'], fontsize=labels['label_fontsize'])
    plt.title(labels['title_1'], fontsize=labels['title_fontsize'])
    plt.yscale('log')
    plt.text(np.max(data['bin_right_squr'])*0.85, np.max(data['count'])*0.55,
             rf"Line fit: ({fit[0]:.2f})$v^2$ + {fit[1]:.2f}" + "\n" + r"$\chi^2_{red} = $" + f"{chisq_red:.4f}",
             fontsize=labels['text_fontsize'], ha='center', va='center')

    # Plotting Residuals
    plt.subplot(*X[1])
    plt.errorbar(data['bin_right_squr'], data['count'] - np.exp(fit[0] * data['bin_right_squr'] + fit[1]),
                 yerr=data['err'],
                 linestyle="None", marker="None", ms=7, ecolor=color[0], mfc=color[1], mew=0, label=labels['label_2'])
    plt.hlines(0, np.min(data['bin_right_squr']), np.max(data['bin_right_squr']), color='black')
    plt.xlabel(labels['xlabel_2'], fontsize=labels['label_fontsize'])
    plt.ylabel(labels['ylabel_2'], fontsize=labels['label_fontsize'])
    plt.title(labels['title_2'], fontsize=labels['title_fontsize'])

    plt.tight_layout()
    plt.savefig(img_path + labels['save_name'])
    plt.show()
    return (fit[0], err_a), chisq_red


# ==============================================================================
# Settings and Running
# ==============================================================================
def main(sets):
    data = read_in(sets['target'])
    data, freq, data_sub, freq_sub = micromotion_sub(sets['target_variable'], data, sets['time_step'],
                                                     sets['region'])
    plot_pos(data, data_sub, freq, freq_sub, sets['time_step'], sets['pos_labels'], sets['img_path'],
             color=sets['colors'])
    velocities = vel_calc(data_sub, sets['time_step'], sets['bins'])
    fit, chi2 = plot_vel(velocities, sets['time_step'], sets['vel_labels'], sets['img_path'], color=sets['colors'])
    print(f"calculated mass: {calc_mass(fit[0]).to(u.pg):.5f} +/- {abs(calc_mass(fit[1]).to(u.pg)):.5f}")
    print(f"chi^2_(red): {chi2:.3f}")


if __name__ == "__main__":
    color0 = ['#3387ec', '#ec9833']

    xscale = 66
    yscale = 67
    xpix = u.def_unit('xpix', 1 * u.mm / xscale)
    ypix = u.def_unit('ypix', 1 * u.mm / yscale)

    pos_labels = {
        'label_fontsize':   20,
        'title_fontsize':   25,
        'legend_fontsize':  15,
        'text_fontsize':    15,


        'data_label_1':     'Initial Data',
        'data_label_2':     r'Inverse FFT (50-80$\rightarrow$0)',
        'data_xlabel':      'Time (s)',
        'data_ylabel':      'Position (mm)',
        'data_title':       'Y Position in Time Domain - 200 Hz',
        'freq_label_1':     'FFT',
        'freq_label_2':     r'FFT (50-80$\rightarrow$0)',
        'freq_xlabel':      'Freq (Hz)',
        'freq_ylabel':      'Amplitude',
        'freq_title':       'Freq Domain - 200 Hz',
        'save_name':        'data_36_y_pos.pdf',
    }
    vel_labels = {
        'label_fontsize':   20,
        'title_fontsize':   25,
        'legend_fontsize':  15,
        'text_fontsize':    15,

        'label_1':          '',
        'xlabel_1':         r'Velocity$^2$ (mm/s)$^2$',
        'ylabel_1':         'Counts',
        'title_1':          'Velocity (Y) Squared - 200 Hz',
        'label_2':          '',
        'xlabel_2':         r'Residuals of Velocity$^2$ (mm/s)$^2$',
        'ylabel_2':         'Counts-Fit',
        'title_2':          'Residuals',
        'save_name':        'data_36_y_vel.pdf',
    }

    settings = {
        'img_path': './images/',
        'pos_labels': pos_labels,
        'vel_labels': vel_labels,
        'colors': color0,

        'target': 'data_36.csv',
        'time_step': 1 / 200,
        'target_variable': 'y',
        'region': (50, 80),
        'bins': 1000,
    }
    main(settings)
