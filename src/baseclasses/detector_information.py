import pathlib
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass()
class Information():
    """
    A dataclass used to encapsulate various information and parameters associated 
    with CycleDetector and post-data processing results.

    Attributes
    ----------
    uniform_data : DataFrame
        The uniformly sampled data after preprocessing.
    event_based_data : DataFrame
        The event-based data derived from the original dataset after preprocessing.
    sample_frequency : int
        The sampling frequency used in the data processing.
    sampling_frequency_unit : int
        The unit for the sampling frequency used during data processing.
    frequency_spectrum_lower_bound : float
        The lower bound for the frequency spectrum.
    frequency_spectrum_upper_bound : float
        The upper bound for the frequency spectrum.
    pyfftw_threshold : int
        The threshold for the pyFFTW algorithm, used in processing the data.
    peak_rate : int
        The rate at which peaks are identified in the data.
    peak_prominence : float
        The prominence of peaks which are identified in the data.
    save_path: pathlib.Path
        The path where the visual results will be saved
    columns : List
        The list of column names in the uniform data.
    """
    uniform_data: pd.DataFrame
    event_based_data: pd.DataFrame
    sample_frequency: int
    sampling_frequency_unit: int
    frequency_spectrum_lower_bound: float
    frequency_spectrum_upper_bound: float
    pyfftw_threshold: int
    peak_rate: int
    peak_prominence: float
    save_path: pathlib.Path
    columns: List
    