import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union


def plot_oneD_fft(x: List[float], 
                 y: List[float], 
                 title: str, 
                 x_lim: Tuple[float, float], 
                 y_lim: float, 
                 x_label: str, 
                 y_label: str, 
                 x_ticks: List[float], 
                 path: Union[str, Path], 
                 variant: int,
                 color: str, 
                 y_lim_2: Optional[float]=None) -> None:
    """
    Plot 1D Fast Fourier Transform either as a scatter plot or as a stem plot.
    
    Parameters:
    - x: List of x-axis values.
    - y: List of y-axis values.
    - title: The title of the plot.
    - x_lim: Tuple containing limits for the x-axis.
    - y_lim: Upper limit for the y-axis.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - x_ticks: Points to be emphasized on the x-axis.
    - path: Directory path where the plot image will be saved.
    - variant: Integer determining the type of plot (0 for scatter, otherwise stem).
    - color: Color for the scatter plot.
    - y_lim_2: Optional bottom limit for the y-axis. If not provided, it will default to the minimum y value.
    
    Returns:
    None. The function saves the plot to the specified directory.
    """
    os.makedirs(path, exist_ok=True)
    size_1 = 40
    size_2 = 30
    size_3 = 20

    if variant == 0:
        plt.figure(figsize=(18,10))
        plt.title(title, fontsize=size_1)
        plt.yticks(fontsize=size_3)
        plt.xticks(x_ticks, rotation=45, fontsize=size_3) # important points to see 
        plt.ylim(top=y_lim, bottom=y_lim_2, auto=True) # set y axis limit (could also set bottom)
        plt.xlim(x_lim) # choose what fits
        plt.scatter(x, y, s=8, c=color) 
        plt.xlabel(x_label, fontsize=size_2) # depends on sample_rate
        plt.ylabel(y_label, fontsize=size_2)
        plt.savefig(path / "scatter.png")
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
        plt.savefig(path / "stem.png")
        plt.close()