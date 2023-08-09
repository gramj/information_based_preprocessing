import pyfftw
import logging
import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks
from numpy.fft import rfft, rfftfreq

from src.monitoring import execution_time_runtime
from src.baseclasses.detector_information import Information
from src.visualization import plot_oneD_fft
from src.baseclasses.detector_output import DetectorOutput

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
    """
    def __init__(self, info: Information):
        super(Detector, self).__init__()
        self.data: np.ndarray = info.uniform_data.values
        
        self.sample_frequency: int = info.sample_frequency
        self.sampling_frequency_unit: int = info.sampling_frequency_unit
        self.frequency_spectrum_lower_bound: float = info.frequency_spectrum_lower_bound
        self.frequency_spectrum_upper_bound: float = info.frequency_spectrum_upper_bound
        self.pyfftw_threshold: int = info.pyfftw_threshold
        
        self.peak_rate: int = info.peak_rate
        self.peak_prominence: float = info.peak_prominence
        
        self.save_path = info.save_path
        self.column_names: List[str] = info.columns

    
    @execution_time_runtime
    def run(self) -> DetectorOutput:
        """
        I/O pass for the Detector: gain information about the cyclic behavior of the input data
        
        Returns
        -------
        plots : plots of the examined data
        result: DetectorOutput =
            cycle_time : float
            cycle_range : Tuple[float, float]
            cyclic_signal_index : int
            cyclic_signal_name : str
            cyclic_signals_indices : Set[int]
            cyclic_signals_names : List[str]
        """
        local_maxima, length = self._peak_detection()
        results = self._prioritizing(data=local_maxima, length=length)
        self._plot_standard(results)
        result = self._compute_output(results=results, length=length)
        return result
      
    def _peak_detection(self) -> Tuple[np.ndarray, int]:
        """
        np.apply_along_axis / np.vectorize not suitable because of varying peak amount per column
        (different shapes)
        
        Returns
        -------
        local_maxima : np.ndarray, col_0 = frequency_value, col_1 = amplitude,
            col_2 = column dataset index for input dataset starting from 0
        length : int, len(self.data)
        """
        length = len(self.data)
        frequency_spectrum, amplitude_array = self._fft_pass(length=length)
        peak_frequencies = []
        peak_amplitudes = []
        peak_indices = []
        distance = length / self.peak_rate
        for i in range(amplitude_array.shape[1]):
            column_amps = amplitude_array[:, i]
            prominence = np.percentile(column_amps, self.peak_prominence)
            peaks, _ = find_peaks(column_amps, distance=distance, prominence=prominence)
            indices = np.repeat(i, len(peaks))
            peak_frequencies.append(frequency_spectrum[peaks])
            peak_amplitudes.append(column_amps[peaks])
            peak_indices.append(indices)
        
        peak_freqs = np.concatenate(peak_frequencies)
        peak_amps = np.concatenate(peak_amplitudes)
        peak_indices = np.concatenate(peak_indices)
        unique_freqs = np.array(list(set(peak_freqs)))
        unique_amps = np.zeros_like(unique_freqs)
        unique_indices = np.zeros_like(unique_freqs)
        for i, freq in enumerate(unique_freqs):
            mask = (peak_freqs == freq)
            amps = peak_amps[mask]
            indices = peak_indices[mask]
            max_index = np.argmax(amps)
            unique_amps[i] = amps[max_index]
            unique_indices[i] = indices[max_index]
        local_maxima = np.concatenate((unique_freqs.reshape(-1, 1), unique_amps.reshape(-1, 1),
                                         unique_indices.reshape(-1, 1)), axis=1)
        return local_maxima, length
        
    
    def _fft_pass(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes a Fast Fourier Transform (FFT) pass which has the highest scaling of computation cost
        in the Detector class.
        
        Parameters
        ----------
        length : int
            Length of the input data for FFT.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the frequency spectrum and amplitude array after performing FFT.

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
        """
        thresholds = [0, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6, np.inf]
        percentiles = [10, 20, 25, 30, 35, 40, 45, 50]
        amplitude_threshold = np.percentile(data[:, 1], percentiles[np.searchsorted(thresholds, length)])
        results = data[data[:, 1] >= amplitude_threshold]
        return results
    
    def _plot_standard(self, result: np.ndarray):
        """
        """
        title_1 = "Frequency domain with weight: (frequency (Hz), amplitude)"
        x_lim = (0.004,0.05)
        y_lim = 100000 
        x_label = "Frequency (Hz)"
        y_label = "Amplitude of frequency"
        x_ticks = [0.004, 0.005, 0.00667, 0.01, 0.0108, 0.0125, 0.022, 0.033, 0.04, 0.05]
        path_1 = self.save_path / "frequency_based_plot"
        plot_oneD_fft(result[:,0], result[:,1], title_1, x_lim, y_lim, x_label, y_label, x_ticks, path_1,
                      variant=1, color="red")

        # convert frequency to seconds (t in s = 1/f in Hz)
        time_from_freq = 1/result[:,0]
        amplitude_t = result[:,1]
        
        title_2 = "Period time with weight: (time (s), amplitude)"
        x_lim = (0,250)
        y_lim = 100000
        x_label = "Period ($seconds$)"
        y_label = "Amplitude of period"
        x_ticks = [10, 20, 30, 45, 60, 80, 89, 93, 97, 100, 120, 150, 180, 200, 225]
        path_1 = self.save_path / "time_based_plot"
        plot_oneD_fft(time_from_freq, amplitude_t, title_2, x_lim, y_lim, x_label, y_label, x_ticks, path_1,
                      variant=1, color="red")
    
    @staticmethod
    def _find_neighbors(arr: np.ndarray, x: float) -> List[float]:
        """
        NOTE sorted array may be useful at other place
        """
        arr = np.sort(arr)
        idx = np.searchsorted(arr, x)
        lower_bound = max(0, idx-3)
        neighbors = arr[lower_bound:idx+3]
        return neighbors
    
    def _compute_output(self, results: np.ndarray, length: int) -> DetectorOutput:
        """
        """
        # cycle_time and strongest cyclic signal
        max_index = np.argmax(results[:, 1])
        cycle_frequency = results[max_index, 0]
        cycle_time = 1 / cycle_frequency
        strongest_cyclic_signal_index = results[max_index, 2].astype(int)
        strongest_cyclic_signal_name = self.column_names[strongest_cyclic_signal_index]
        
        # cycle_range
        neighbor_values = self._find_neighbors(arr=results[:,0], x=cycle_frequency)
        lower_freq_limit = neighbor_values[0]
        upper_freq_limit = neighbor_values[-1]
        lower_time_limit = 1 / upper_freq_limit
        upper_time_limit = 1 / lower_freq_limit
        
        # cyclic_signals
        thresholds = [0, 5e5, 1e6, 2e6, 3e6, 4e6, 5e6, np.inf]
        percentiles = [40, 70, 75, 85, 91, 95, 96, 97]
        top_amp = np.percentile(results[:,1], percentiles[np.searchsorted(thresholds, length)])
        indices = np.where(results[:,1] >= top_amp)[0]
        cyclic_signals_indices = results[indices,2].astype(int)
        cyclic_signals_indices = set(cyclic_signals_indices)
        cyclic_signals_names = np.array(self.column_names)[list(cyclic_signals_indices)].tolist()
        
        result = DetectorOutput(
            cycle_time=cycle_time,
            cycle_range=(lower_time_limit, upper_time_limit),
            cyclic_signal_index=strongest_cyclic_signal_index,
            cyclic_signal_name=strongest_cyclic_signal_name,
            cyclic_signals_indices=cyclic_signals_indices,
            cyclic_signals_names=cyclic_signals_names
        )
        return result

