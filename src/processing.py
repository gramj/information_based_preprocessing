import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sympy import isprime

from src.baseclasses.processing_parameters import ProcessingParameters
from src.feature_selection import(
    correlation_filter,
    low_variance_filter
)

logger = logging.getLogger(__name__)


def data_preprocessing(data: pd.DataFrame, params: ProcessingParameters
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    processing data for cycle detection:
    1. removal of low variance columns
    2. removal of very high correlating columns
    3. uniformizing the datasets frequency for event-based datasets
    """ 
    first_column = data.iloc[:, 0]
    data = data.drop(data.columns[0], axis=1)
    logger.info(f"raw {data.shape=}")
        
    indices = low_variance_filter(data=data, dynamic=params.dynamic_variance,
                                  dynamic_threshold=params.dynamic_threshold)
    low_variance_cols = data.columns[list(indices)]
    high_correlating_cols = correlation_filter(data=data,
                                               correlation_threshold=params.correlation_threshold)
    drop_columns = list(low_variance_cols) + high_correlating_cols
    data = data.drop(columns=drop_columns)

    logger.info(f"preprocessed {data.shape=}")
    
    data.insert(0, "timestamp", first_column)
    data["timestamp"]= pd.to_datetime(data["timestamp"], unit='ns')
    data = data.set_index("timestamp")
    data = data.sort_index()
    event_based_data = data.copy()

    frequency = compute_sampling_frequency(data=data,
                                           sampling_frequency_unit=params.sampling_frequency_unit,
                                           sampling_frequency_round=params.sampling_frequency_round)
    data = data.asfreq(freq=str(frequency) + "ms", method="ffill")
    data.reset_index(inplace=True, drop=True)
    return data, event_based_data, frequency


def compute_sampling_frequency(data: pd.DataFrame, sampling_frequency_unit: int=1e3,
                               sampling_frequency_round: int=50) -> int:
    """
    Compute the frequency (in millihertz) at which data was recorded or events were triggered 
    on average in a time series data. This is achieved by first determining the average duration 
    (in seconds) between consecutive data points, then inverting this value to get the number 
    of samples per second (frequency), and finally converting this frequency to millihertz.

    The computed frequency is then rounded to the nearest value that's a multiple of a specified 
    round number. This helps in making the frequency more manageable and in certain cases can 
    improve the performance of further computations that use this frequency.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of 2D event-based time series data. Its index should represent time.

    sampling_frequency_unit : int, optional
        The unit to convert the computed frequency into. This is effectively a scaling factor. 
        Default is 1e3 (millihertz).

    sampling_frequency_round : int, optional
        The number to which the computed frequency should be rounded. The final frequency 
        is rounded to the nearest multiple of this number. Default is 50.

    Returns
    -------
    frequency : int
        The computed, scaled, and rounded frequency of the data in the specified unit.
    """
    dataset_timeframe = data.index[-1] - data.index[0]
    dataset_timeframe_seconds = dataset_timeframe.total_seconds()
    num_samples = len(data)
    x = dataset_timeframe_seconds / num_samples
    frequency = sampling_frequency_unit / x
    frequency = int(np.round(frequency / sampling_frequency_round) * sampling_frequency_round)
    return frequency


def data_postprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply postprocessing to the input data by truncating its length.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be processed

    Returns
    -------
    data : pd.DataFrame 
        The processed data
    """
    data = truncate_length(data=data)
    return data


def truncate_length(data: pd.DataFrame) -> pd.DataFrame:
    """
    Truncate the length of the input data to improve the performance of FFT approximation algorithms.
    These algorithms generally perform faster when len(data) is not a prime number and is a power of 2 (4, 8).

    Parameters
    ----------
    data : The data to be reduced row-wise

    Returns
    -------
    truncated_data : The truncated data
    """
    new_length = int(2 ** np.floor(np.log2(len(data))))
    if new_length >= 0.67 * len(data):
        truncated_data = data[:new_length]
    else:
        if isprime(len(data)):
            truncated_data = data.iloc[:-1]
        else:
            truncated_data = data
    return truncated_data