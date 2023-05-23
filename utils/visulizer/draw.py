import random

import matplotlib.pyplot as plt


def save_plot(data, xlabel, ylabel, save_path):

    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
