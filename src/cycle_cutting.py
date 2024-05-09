import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.monitoring import execution_time_runtime

logger = logging.getLogger(__name__)


@execution_time_runtime
def cut_cycles(data: pd.DataFrame, primary_signal: str, true_cycle_time: float, secondary_signal: str=None,
               edge_condition: int=1, version: int=0, merge_threshold: float=0.65, min_rows_per_cycle: int=9
               ) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Detects cycles based on two cyclic signals within the data and calculates the duration of each cycle.
    It assigns a cycle index to each row of the DataFrame and returns the modified DataFrame and a dictionary of cycle durations.
    This function optimizes cycle detection by reducing DataFrame copies and leveraging efficient pandas operations.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing timestamped data with a DateTime index.
    primary_signal : str
        The column name in 'data' which contains the primary cyclic signal.
    secondary_signal : str
        The column name in 'data' which contains the secondary cyclic signal.
    true_cycle_time : float
        Expected cycle time, used as a reference for merging short cycles.
    edge_condition : int, optional
        Condition for detecting cycle starts (1 for rising edges, -1 for falling edges).
    merge_threshold : float
        Threshold for merging cycles, relative to the true cycle time.
    version : int
        Version flag to determine the logic for combining signals (0 for primary only, 1 for primary OR secondary).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, float]]
        A modified DataFrame with cycle numbers and a dictionary mapping cycle numbers to their durations.
    """
    primary_diff = data[primary_signal].diff()
    primary_starts = (primary_diff == edge_condition)
    combined_cycle_starts = primary_starts
    
    if version == 1 and secondary_signal is not None:
        secondary_diff = data[secondary_signal].diff()
        secondary_starts = (secondary_diff == edge_condition)
        combined_cycle_starts = primary_starts | secondary_starts
    cycle_indices = combined_cycle_starts.cumsum()
    cycled_data = data.copy()
    cycled_data["cycle_number"] = cycle_indices

    cycle_start_indices = np.flatnonzero(cycled_data["cycle_number"].diff().gt(0))
    if not cycle_start_indices.any():
        return cycled_data.iloc[0:0], {}
    # Adjusting indices to start with the first full cycle
    cycled_data = cycled_data.iloc[cycle_start_indices[0]:]
    cycled_data["cycle_number"] -= cycled_data["cycle_number"].iloc[0] - 1
    cycle_start_indices -= cycle_start_indices[0]
    # Calculating cycle times
    cycle_times = (cycled_data.index[cycle_start_indices] - cycled_data.index[np.r_[0, cycle_start_indices[:-1]]]).total_seconds()
    cycle_times_dict = {}
    new_cycle_numbers = []
    valid_cycle_indices = []

    for i, start_idx in enumerate(cycle_start_indices):
        if i == 0 or cycle_times[i - 1] >= true_cycle_time * merge_threshold:
            valid_cycle_indices.append(start_idx)
            new_cycle_numbers.append(len(new_cycle_numbers) + 1)
        else:
            cycled_data.loc[cycled_data.index[cycle_start_indices[i - 1]]:cycled_data.index[cycle_start_indices[i]], 'cycle_number'] = new_cycle_numbers[-1]

    unique_cycles = sorted(cycled_data['cycle_number'].unique())
    cycle_mapping = {old_num: new_num for new_num, old_num in enumerate(unique_cycles, start=1)}
    cycled_data['cycle_number'] = cycled_data['cycle_number'].map(cycle_mapping)

    # Build cycle times dictionary without the last cycle
    for i in range(1, len(valid_cycle_indices)):
        start_idx = valid_cycle_indices[i - 1]
        end_idx = valid_cycle_indices[i] - 1
        duration = (cycled_data.index[end_idx] - cycled_data.index[start_idx]).total_seconds()
        cycle_times_dict[new_cycle_numbers[i - 1]] = duration
    # Remove last cycle from the data and dictionary if it exists
    if new_cycle_numbers:
        last_cycle_number = new_cycle_numbers[-1]
        if valid_cycle_indices[-1] < len(cycled_data) - 1:
            cycled_data = cycled_data[cycled_data["cycle_number"] < last_cycle_number]
            cycle_times_dict.pop(last_cycle_number, None)


    cycle_times_dict = {}
    new_cycle_numbers = []
    valid_cycle_indices = []
    cycle_start_indices = np.flatnonzero(cycled_data["cycle_number"].diff().gt(0))
    cycle_row_counts = cycled_data.groupby('cycle_number').size().reset_index(drop=True)
    for i, start_idx in enumerate(cycle_start_indices):
        if i == 0 or cycle_row_counts[i - 1] >= min_rows_per_cycle:
            valid_cycle_indices.append(start_idx)
            new_cycle_numbers.append(len(new_cycle_numbers) + 1)
        else:
            cycled_data.loc[cycled_data.index[cycle_start_indices[i - 1]]:cycled_data.index[cycle_start_indices[i]], 'cycle_number'] = new_cycle_numbers[-1]
    # Reassign cycle numbers sequentially
    unique_cycles = sorted(cycled_data['cycle_number'].unique())
    cycle_mapping = {old_num: new_num for new_num, old_num in enumerate(unique_cycles, start=1)}
    cycled_data['cycle_number'] = cycled_data['cycle_number'].map(cycle_mapping)
    # Build cycle times dictionary without the last cycle
    for i in range(1, len(valid_cycle_indices)):
        start_idx = valid_cycle_indices[i - 1]
        end_idx = valid_cycle_indices[i] - 1
        duration = (cycled_data.index[end_idx] - cycled_data.index[start_idx]).total_seconds()
        cycle_times_dict[new_cycle_numbers[i - 1]] = duration
    # Remove last cycle from the data and dictionary if it exists
    if new_cycle_numbers:
        last_cycle_number = new_cycle_numbers[-1]
        if valid_cycle_indices[-1] < len(cycled_data) - 1:
            cycled_data = cycled_data[cycled_data["cycle_number"] < last_cycle_number]
            cycle_times_dict.pop(last_cycle_number, None)
    return cycled_data, cycle_times_dict