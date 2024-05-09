import pathlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_fft(x: np.ndarray, y: np.ndarray, title: str, x_lim: Tuple[int],
             y_lim: int, x_label: str, y_label: str, x_ticks: List[int],
             path: pathlib.Path, data_variant: str, plot_variant: int=1,
             color: str='red', y_lim_2: int=None) -> None:
    """
    """
    plt.rcParams.update({
        'axes.facecolor': 'black',
        'savefig.facecolor': 'black',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'font.size': 18
    })
    
    size_1 = 40
    size_2 = 30
    size_3 = 20

    if plot_variant == 0:
        plt.figure(figsize=(18,10))
        plt.title(title, fontsize=size_1)
        plt.yticks(fontsize=size_3)
        plt.xticks(x_ticks, rotation=45, fontsize=size_3) # important points to see 
        plt.ylim(top=y_lim, bottom=y_lim_2, auto=True) # set y axis limit (could also set bottom)
        plt.xlim(x_lim) # choose what fits
        plt.scatter(x, y, s=8, c=color) 
        plt.xlabel(x_label, fontsize=size_2) # depends on sample_rate
        plt.ylabel(y_label, fontsize=size_2)
        plt.savefig(path / f"{data_variant}_scatter.png")
        plt.close()
    else:
        plt.figure(figsize=(18,10))
        plt.title(title, fontsize=size_1)
        plt.yticks(fontsize=size_3)
        plt.xticks(x_ticks, rotation=45, fontsize=size_3)
        plt.ylim(top=y_lim, bottom=y_lim_2, auto=True)
        plt.xlim(x_lim)
        plt.stem(x, y, linefmt='C3-')
        plt.xlabel(x_label, fontsize=size_2)
        plt.ylabel(y_label, fontsize=size_2)
        plt.savefig(path / f"{data_variant}_stem.png")
        plt.close()