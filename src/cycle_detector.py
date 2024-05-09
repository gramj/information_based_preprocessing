import logging
import pathlib
import pyfftw
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from typing import List, Tuple

from src.baseclasses.detector_information import Information
from src.baseclasses.detector_output import DetectorOutput
from src.monitoring import execution_time_runtime
from src.visualization import plot_fft

logger = logging.getLogger(__name__)

    
class Detector:
    """  
    Detect cyclical behavior in 2D time series data arrays in terms of:
     - cycle time
     - cycle time range
     - cyclic signals
     
    Parameters
    ----------
    info : An object of Information data class
    
    Information
    -----------
    The Detector class aims to detect cyclical behavior in 2D time series data. The key functionalities include:
    1. Performing a Fast Fourier Transform (FFT) to transform the data into the frequency domain.
    2. Detecting peaks in the frequency domain to identify cycle time and cyclic signals.
    3. Prioritizing peaks based on their amplitudes.
    4. Computing and returning a DetectorOutput object that encapsulates all computed parameters.
    
    Risks
    -----
    Data Risks:
        Input data must be clean and uniformly sampled for accurate results. Outliers, noise or missing values
            may affect the performance, as well as inaccurate Sampling Frequency.
        Incorrect parameter settings may also result in unreliable output ->
            Parameters like `pyfftw_threshold`, `peak_rate`, and `peak_prominence` may need tuning.
        Computational Limitations: Very large datasets could lead to computational inefficiency.
    Analytical Risks:
        The FFT and peak detection algorithms are not foolproof and can produce false positives or negatives.
        The cycle time and cycle signals are inferred, not guaranteed.
    """
    def __init__(self, info: Information, version: int=0):
        super(Detector, self).__init__()
        self.data: np.ndarray = info.uniform_data.values
        self.cycle_version = version
        
        self.sample_frequency: int = info.sample_frequency
        self.sampling_frequency_unit: int = info.sampling_frequency_unit
        self.frequency_spectrum_lower_bound: float = info.frequency_spectrum_lower_bound
        self.frequency_spectrum_upper_bound: float = info.frequency_spectrum_upper_bound
        self.pyfftw_threshold: int = info.pyfftw_threshold
        
        self.peak_rate: int = info.peak_rate
        self.peak_prominence: float = info.peak_prominence
        
        self.column_names: List[str] = info.columns
        self.save_path: pathlib.Path = info.data_path
    
    @execution_time_runtime
    def run(self) -> DetectorOutput:
        """
        I/O pass for the Detector: gain information about the cyclic behavior of the input data
        
        Returns
        -------
        result: DetectorOutput =
            cycle_time : float
            cycle_range : Tuple[float, float]
            cyclic_signal_index : int
            cyclic_signal_name : str
            cyclic_signals_indices : Set[int]
            cyclic_signals_names : List[str]
        
        Information
        -----------
        This is the main entry point for running the detector. It orchestrates the following:
        1. Peak Detection: Identifies frequency peaks based on amplitude.
        2. Prioritizing: Filters the peaks based on amplitude percentiles.
        3. Output Computation: Aggregates the computed parameters into a DetectorOutput object.
        
        Risks
        -----
        Data Risks:
            The output can be incorrect if the underlying data contains noise, missing values,
                or if it is not uniformly sampled.
        Analytical Risks:
            Assumes that the peak with the highest amplitude corresponds to the most significant cyclic behavior,
                which may not always be true.
            If the peak prominence and rate are not tuned correctly, it may lead to incorrect cycle times and signals.
        """
        local_maxima, length = self._peak_detection()
        results = self._prioritizing(data=local_maxima, length=length)
        self._plot_fft(results)
        result = self._compute_output(results=results, length=length)
        return result
    
    @execution_time_runtime
    def _peak_detection(self) -> Tuple[np.ndarray, int]:
        """
        NOTE bottleneck O(n x m) due to peak detection part
        np.apply_along_axis / np.vectorize not suitable because of varying peak amount per column
        (different shapes)
        
        Returns
        -------
        local_maxima : np.ndarray, col_0 = frequency_value, col_1 = amplitude,
            col_2 = column dataset index for input dataset starting from 0
        length : int, len(self.data)
        
        Information
        -----------
        The '_peak_detection' method uses the Fast Fourier Transform (FFT) and peak detection algorithms to 
        identify the local maxima in the frequency domain for each column of the dataset. 
        
        Risks
        -----
        Data Risks:
            Incorrect peak prominence and distance settings can result in false positives or negatives.
        Analytical Risks:
            The method assumes that all peaks are equally important across different columns,
                which might not be the case.
            Low prominence value could miss relevant peaks.
        """
        length = len(self.data)
        frequency_spectrum, amplitude_array = self._fft_pass(length=length)
        distance = length / self.peak_rate
        prominence_values = np.percentile(amplitude_array, self.peak_prominence, axis=0)
        peak_frequencies, peak_amplitudes, peak_indices = [], [], []
        for i, column_amps in enumerate(amplitude_array.T): 
            peaks, _ = find_peaks(column_amps, distance=distance, prominence=prominence_values[i])
            peak_frequencies.extend(frequency_spectrum[peaks])
            peak_amplitudes.extend(column_amps[peaks])
            peak_indices.extend(np.repeat(i, len(peaks)))
        
        structured_arr = np.zeros(len(peak_frequencies), dtype={'names':('freq', 'amp', 'idx'),
                                                           'formats':(float, float, int)})
        structured_arr['freq'] = peak_frequencies
        structured_arr['amp'] = peak_amplitudes
        structured_arr['idx'] = peak_indices
        unique_freqs = np.unique(structured_arr['freq'])
        max_unique_amps = []
        max_unique_idxs = []

        for uf in unique_freqs:
            mask = structured_arr['freq'] == uf
            max_amp_idx = np.argmax(structured_arr['amp'][mask])
            max_unique_amps.append(structured_arr['amp'][mask][max_amp_idx])
            max_unique_idxs.append(structured_arr['idx'][mask][max_amp_idx])
        local_maxima = np.vstack((unique_freqs, max_unique_amps, max_unique_idxs)).T
        return local_maxima, length
        
    def _fft_pass(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        NOTE
        # Assuming 'data' is a 2D dataset where each row is a sensor's time series
        frequencies_2d = np.fft.fft2(data)
        amplitude_2d = np.abs(frequencies_2d)

        # Removing the DC component
        amplitude_2d[0, 0] = 0

        # Find the position of the maximum value in the 2D spectrum
        dominant_freq_position = np.unravel_index(np.argmax(amplitude_2d, axis=None), amplitude_2d.shape)

        # This position will correspond to a combination of frequencies in time and across sensors
        time_freq = dominant_freq_position[1] / data.shape[1]  # assuming your data is sampled uniformly in time
        sensor_freq = dominant_freq_position[0] / data.shape[0]  # assuming sensors are uniformly spaced or indexed
                Executes a Fast Fourier Transform (FFT) pass which has the highest scaling of computation cost in the Detector class.
        
        
        Parameters
        ----------
        length : int
            Length of the input data for FFT.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the frequency spectrum and amplitude array after performing FFT.

        Information
        -----------
        This method performs Fast Fourier Transform on the input data. It decides whether to use Numpy's `rfft` or 
        pyFFTW's `rfft` based on the size of the data. The frequency spectrum is then filtered between user-defined
        lower and upper bounds.
        
        Risks
        -----
        1. Computation Time: If the data set is too large, the FFT operation could be computationally expensive.
        2. Frequency Resolution: The lower and upper bounds for frequency might filter out relevant data.
        3. Library Differences: Different FFT implementations (Numpy vs pyFFTW) could yield subtly different results.
        
        Notes
        -----
        The FFT pass is done using numpy's rfft if the product of length and number of columns in data is 
        less than or equal to a threshold, otherwise pyFFTW's rfft is used.
        """
        sample_rate = self.sampling_frequency_unit / self.sample_frequency # milliseconds 
        frequency_spectrum = rfftfreq(length, d=1.0 / sample_rate)
        first_index = np.argmax(frequency_spectrum > self.frequency_spectrum_lower_bound)
        last_index = np.argmax(frequency_spectrum > self.frequency_spectrum_upper_bound)
        frequency_spectrum = frequency_spectrum[first_index:last_index]
        if length * self.data.shape[1] <= self.pyfftw_threshold:
            amplitude_array = rfft(self.data, axis=0)
        else:
            amplitude_array = pyfftw.interfaces.numpy_fft.rfft(self.data, axis=0)
        amplitude_array = np.abs(amplitude_array)
        amplitude_array = amplitude_array[first_index:last_index, :]
        return frequency_spectrum, amplitude_array
    
    @staticmethod
    def _prioritizing(data: np.ndarray, length: int) -> np.ndarray:
        """
        choose the top amplitude values by percentile (top x% of values)
        
        Parameters
        ----------
        data : np.ndarray
            The 2D array containing frequency spectrum data, analyzed by local extrema.
            Each row represents a data point [frequency, amplitude].
        length: int
            data.shape[0]
        
        Returns
        -------
        results : np.ndarray
            A filtered 2D array containing only the top amplitude values based on percentile thresholds.
        
        Information
        -----------
        Filters the frequency-amplitude data by selecting only the top x% of amplitude values based
        on a predefined percentile.
        
        Risks
        -----
        1. Amplitude Cut-off: The percentile cut-off might eliminate relevant peaks.
        2. Sensitivity to Length: Different data lengths might require different percentile thresholds
            for effective filtering.
        """
        thresholds = [0, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6, np.inf]  # TODO config
        percentiles = [10, 20, 25, 30, 35, 40, 45, 50]
        amplitude_threshold = np.percentile(data[:, 1], percentiles[np.searchsorted(thresholds, length)])
        results = data[data[:, 1] >= amplitude_threshold]
        return results
    
    def _plot_fft(self, result: np.ndarray) -> None:
        """
        """
        title_1 = "Frequency domain with weight: (frequency (Hz), amplitude)"
        x_lim = (0.0004,0.05) # NOTE
        # x_lim = (0.0004,0.2)
        y_lim = 100000 
        x_label = "Frequency (Hz)"
        y_label = "Amplitude of frequency"
        x_ticks = [0.004, 0.005, 0.00667, 0.01, 0.0108, 0.0125, 0.022, 0.033, 0.04, 0.05] # NOTE
        plot_fft(result[:,0], result[:,1], title_1, x_lim, y_lim, x_label, y_label, x_ticks, self.save_path,
                      data_variant='frequency', plot_variant=1, color="red")

        # convert frequency to seconds (t in s = 1/f in Hz)
        time_from_freq = 1/result[:,0]
        amplitude_t = result[:,1]
        
        title_2 = "Period time with weight: (time (s), amplitude)"
        x_lim = (0,250)
        y_lim = 100000
        x_label = "Period ($seconds$)"
        y_label = "Amplitude of period"
        x_ticks = [15, 20, 30, 45, 60, 80, 89, 93, 97, 100, 120, 150, 180, 200, 225]
        plot_fft(time_from_freq, amplitude_t, title_2, x_lim, y_lim, x_label, y_label, x_ticks, self.save_path,
                      data_variant='time', plot_variant=1, color="red")
        
    @staticmethod
    def _find_neighbors(arr: np.ndarray, x: float, distance: int=3) -> np.ndarray:
        """
        Finds the neighboring elements around a given value in a sorted array.
        
        Parameters
        ----------
        arr : np.ndarray
            The sorted 1D array to search within.
        x : float
            The value around which to find neighbors.
        distance: int
            The neighboring area TODO config
            
        Returns
        -------
        neighbors : np.ndarray
            An array of neighboring elements around the given value.
        
        Risks
        -----
        1. Boundary Cases: If the target value `x` is near the edges of the array, the number of neighbors
            might be less than expected.
        2. Configuration Sensitivity: The distance parameter for neighbors (e.g., `idx-3` and `idx+3`)
            could require tuning based on data characteristics.
        """
        idx = np.searchsorted(arr, x)
        lower_bound = max(0, idx-distance)
        upper_bound = min(len(arr), idx+distance)
        return arr[lower_bound:upper_bound]
        
    def _compute_output(self, results: np.ndarray, length: int) -> DetectorOutput:
        """
        Computes the output of the detector given frequency spectrum data.
        
        Parameters
        ----------
        results : np.ndarray
            The 2D array containing frequency spectrum data. 
            Each row represents a data point [frequency, amplitude, signal_index].
        length : int
            The length of the data array (i.e., results.shape[0]).
            
        Returns
        -------
        DetectorOutput
            An object containing various computed output parameters such as
            cycle_time, cycle_range, and strongest cyclic signals.
        """      
        sorted_results = results[results[:, 1].argsort()[::-1]]
        # cycle_time and strongest cyclic signal
        top_frequency = sorted_results[0, 0]
        top_index = sorted_results[0, 2]
        remaining_results = sorted_results[sorted_results[:, 2] != top_index]
        if len(remaining_results) > 1:
            second_top_frequency = remaining_results[1, 0]
        else:
            second_top_frequency = sorted_results[1, 0]
        if self.cycle_version == 0:
            cycle_frequency = top_frequency
        else:
            cycle_frequency = 0.6 * top_frequency + 0.4 * second_top_frequency
        cycle_time = 1 / cycle_frequency
        most_cyclic_signals = [sorted_results[i, 2].astype(int) for i in range(2)]
        
        # cycle_range
        neighbor_values = self._find_neighbors(arr=sorted_results[:, 0], x=cycle_frequency, distance=3)
        lower_freq_limit = neighbor_values[0]
        upper_freq_limit = neighbor_values[-1]
        lower_time_limit = 1 / upper_freq_limit
        upper_time_limit = 1 / lower_freq_limit
        cycle_range = (lower_time_limit, upper_time_limit)
        
        # cyclic_signals
        thresholds = [0, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6, np.inf] # TODO config
        percentiles = [40, 70, 75, 85, 91, 95, 96, 97]
        top_amp = np.percentile(results[:,1], percentiles[np.searchsorted(thresholds, length)])
        indices = np.where(results[:,1] >= top_amp)[0]
        cyclic_signals_indices = results[indices,2].astype(int)
        cyclic_signals_indices = set(cyclic_signals_indices)
        cyclic_signals_names = np.array(self.column_names)[list(cyclic_signals_indices)].tolist()
        
        result = DetectorOutput(
            cycle_time=cycle_time,
            cycle_range=cycle_range,
            primary_cyclic_signal_index=most_cyclic_signals[0],
            primary_cyclic_signal_name=self.column_names[most_cyclic_signals[0]],
            secondary_cyclic_signal_index=most_cyclic_signals[1],
            secondary_cyclic_signal_name=self.column_names[most_cyclic_signals[1]],
            cyclic_signals_indices=cyclic_signals_indices,
            cyclic_signals_names=cyclic_signals_names
        )
        return result


