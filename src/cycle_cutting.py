import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.monitoring import execution_time_runtime
    
    
@execution_time_runtime
def cut_cycles(data: pd.DataFrame, cyclic_signal: str, true_cycle_time: float,
               edge_condition: int=1) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    The function identifies the start of a new cycle based on a defined feature edge condition
    and assigns a cycle index to each row of the DataFrame.
    
    We consider detected cycles which are significantly shorter than the calculated true cycle time as data anomalies.  
    Hence we ignore the flank for these cases.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the timestamped data with a DateTime index.
    cyclic_signal : str
        The name of the column in 'data' which contains the cyclic signal.
    true_cycle_time : float
        The expected duration of a cycle in seconds. Used as a reference for anomaly detection.
    edge_condition : int, optional
        Specifies the condition for determining the start of a cycle based on the transitions in 'cyclic_signal'.
        If 1, the function considers the rising edges (transitions from 0 to 1) as the start of a cycle.
        If -1, the function considers the falling edges (transitions from 1 to 0) as the start of a cycle.
        Default is 1.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, float]]
        A tuple containing two elements:
        
        1. DataFrame: This is a modified version of 'data' with an additional 'cycle_number' column
        at the last index which  contains the cycle number per row.
        
        2. Dictionary: Each key in this dictionary is a cycle number, and the corresponding value is the cycle 
        duration in seconds.
    """
    data_diff = data[cyclic_signal].diff()

    if edge_condition == 1:
        cycle_starts = (data_diff == 1)
    elif edge_condition == -1:
        cycle_starts = (data_diff == -1)
    cycle_indices = cycle_starts.cumsum()
    
    cycled_data = data.copy()
    cycled_data["cycle_number"] = cycle_indices

    cycle_changes = cycled_data['cycle_number'].values[:-1] != cycled_data['cycle_number'].values[1:]
    cycle_start_timestamps = cycled_data.index.values[np.r_[cycle_changes, True]]
    cycle_times = (cycle_start_timestamps[1:-1] - cycle_start_timestamps[:-2])
    cycle_times = pd.Series(cycle_times).dt.total_seconds()
    new_cycle_number = cycle_indices.copy()
    cycle_anomaly_counter = 0
    cycle_start_indices = [0]
    for i in range(1, len(cycle_indices)):
        if cycle_indices[i] > cycle_indices[i-1]:
            cycle_time = data.index[i-1] - data.index[cycle_start_indices[-1]]
            if cycle_time.total_seconds() < true_cycle_time * 0.75:
                cycle_anomaly_counter += 1
                if cycle_anomaly_counter > 0:
                    cycle_time_next = data.index[i] - data.index[cycle_start_indices[-1]]
                    if cycle_time_next.total_seconds() >= true_cycle_time * 0.75:
                        new_cycle_number[cycle_indices > cycle_indices[cycle_start_indices[-1]]] -= \
                            cycle_anomaly_counter
                        cycle_anomaly_counter = 0
            else:
                if cycle_anomaly_counter > 0:
                    new_cycle_number[cycle_indices > cycle_indices[cycle_start_indices[-1]]] -= \
                        cycle_anomaly_counter
                    cycle_anomaly_counter = 0
            if not cycle_anomaly_counter:
                cycle_start_indices.append(i)
    if cycle_anomaly_counter > 0:
        new_cycle_number[cycle_indices > cycle_indices[cycle_start_indices[-1]]] -= cycle_anomaly_counter

    cycled_data["cycle_number"] = new_cycle_number
    new_cycle_start_timestamps = cycled_data.index[cycle_start_indices]
    new_cycle_times = (new_cycle_start_timestamps[1:] - new_cycle_start_timestamps[:-1])
    new_cycle_times = pd.Series(new_cycle_times).dt.total_seconds()
    cycle_times_dict = {i+1: new_cycle_times[i] for i in range(len(new_cycle_times))}
    number_of_complete_cycles = len(new_cycle_times)
    cycled_data = cycled_data[(cycled_data['cycle_number'] != 0) & (cycled_data['cycle_number'] \
        <= number_of_complete_cycles)]
    return cycled_data, cycle_times_dict