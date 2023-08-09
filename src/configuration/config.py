import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser_ibp():
    parser = argparse.ArgumentParser()
    
    # --- Data params ----
    parser.add_argument("--data_set", type=str, default="PLC_data_1",
    help='''name of the dataset''')
    parser.add_argument("--separator", type=str, default=",")
    
    
    # --- Preprocessing params ----
    parser.add_argument("--dynamic_variance", type=str2bool, default=False,
                        help="compute filtering threshold dynamically")
    parser.add_argument("--dynamic_threshold", type=float, default=2.05,
                        help='''float to multiply the median absolute deviation bevor substracting
                        from median of variances. dynamic_threshold: 2.0, 2.03, 2.04, 2.045.''')
    parser.add_argument("--correlation_threshold", type=float, default=0.98,
    help='''if signal correlation >= correlation_threshold -> high correlating feature
    -> removal process''')
    
    parser.add_argument("--gamma_threshold", type=float, default=0.5,
    help='''percentile value. AutoCorrelation values of signals will be considered acyclic
    if they fall below this percentile of the fit Gamma distribution.''')
    
    parser.add_argument("--sampling_frequency_unit", type=int, default=1e3,
    help='''conversion unit for the sampling frequencybased on sample rate in seconds.
    1e3 = millihertz, 1 = hertz''')
    parser.add_argument("--sampling_frequency_round", type=int, default=50,
    help="round the calculated data sampling frequency to nearest multiple of round_sampling_frequency")
    

    # Cycle Detection params ----
    parser.add_argument("--frequency_spectrum_lower_bound", type=float, default=0.004,
    help="frequency values below this threshold will be cut - cycle time > 250s")
    parser.add_argument("--frequency_spectrum_upper_bound", type=float, default=0.125,
    help="frequency values above this threshold will be cut - cycle time < 8s")
    parser.add_argument("--pyfftw_threshold", type=int, default=1e8,
    help='''datasets with more data points than this threshold will be processed
    with the pyfftw fft implementation''')
    
    parser.add_argument("--peak_rate", type=int, default=2000,
    help='''len(data) / peak_rate = distance -> Required minimal horizontal distance (>= 1) in samples
    between neighbouring peaks for find_peaks''')
    parser.add_argument("--peak_prominence", type=float, default=0.85,
    help="percentile threshold for thee required prominence of peaks in find_peaks")
    
    
    # --- Cycle Cutting params ----
    parser.add_argument("--cyclic_signal_edge", type=int, default=1,
                        help='''Behavior of main cyclic signal 
                        -1: falling edge (Flanke) (1 -> 0);
                        1: rising edge (Flanke) (0 -> 1);
                        ''')

    return parser
