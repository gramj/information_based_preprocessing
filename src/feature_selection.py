import logging
import numpy as np
import pandas as pd
from scipy.stats import gamma
from typing import List, Set, Union

logger = logging.getLogger(__name__)


def low_variance_filter(data: Union[pd.DataFrame, np.ndarray],
                             dynamic: bool=True, dynamic_threshold: float=0.1) -> Set[int]:
    """
    Calculate the variance per column in a given dataset. If 'dynamic' is set to True,
    fit a gamma distribution to these variances, and return indices of columns whose 
    variance is below the dynamic_threshold percentile of the gamma distribution. 
    If 'dynamic' is False, return the indices of columns that have low absolute variance.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset to compute variances on.
    dynamic : bool
        If True, uses a dynamic threshold based on a gamma distribution fitted
        to the variances. If False, returns columns with zero variance.
        Default is True.
    dynamic_threshold : float
        The percentile of the gamma distribution to use as a threshold when 
        'dynamic' is True.

    Returns
    -------
    low_variance_cols : set
        Set of indices of the columns whose variance is below the dynamic_threshold percentile
        of the gamma distribution if 'dynamic' is True, or the indices of the columns 
        with zero variance if 'dynamic' is False.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    variances = np.var(data, axis=0)
    
    if dynamic:
        variances = (variances - np.min(variances)) / (np.max(variances) - np.min(variances))
        shape, loc, scale = gamma.fit(variances)
        percentile = gamma.ppf(dynamic_threshold, shape, loc, scale)
        low_variance_cols = set(np.where(variances < percentile)[0])
    else:
        low_variance_cols = set(np.where(variances == 0.02)[0])
    return low_variance_cols


def correlation_filter(data: pd.DataFrame, correlation_threshold: float=0.95) -> List[str]:
    """
    Computes a correlation matrix and returns columns with high correlation to be dropped.

    Parameters
    ----------
    data : pd.DataFrame
        Processed timeseries data of signal columns.
    correlation_threshold : float, optional
        Threshold for strong correlation between features (corr >= threshold will be dropped).

    Returns
    -------
    drop_cols : List[str]
        List of column names to be dropped based on high correlation with other columns.
    """
    variances = np.var(data.values, axis=0)
    corr_mat = np.corrcoef(data.values, rowvar=False)
    np.fill_diagonal(corr_mat, 0.0)

    high_corr_indices = np.where(corr_mat >= correlation_threshold)

    drop_cols = set()
    for i, j in zip(*high_corr_indices):
        if variances[i] > variances[j]:
            drop_cols.add(data.columns[j])
        else:
            drop_cols.add(data.columns[i])
    return list(drop_cols)