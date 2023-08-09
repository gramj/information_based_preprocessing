from dataclasses import dataclass


@dataclass()
class ProcessingParameters():
    """
    A dataclass used to encapsulate various parameters used for feature selection 
    and upsampling operations during data processing.
    """
    dynamic_variance: bool
    dynamic_threshold: float
    correlation_threshold: float
    gamma_threshold: float
    data_name: str
    sampling_frequency_unit: int
    sampling_frequency_round: int

    