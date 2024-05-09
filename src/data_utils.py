import logging
import pathlib
import ujson
import pandas as pd
from datetime import datetime

from src.baseclasses.detector_information import Information
from src.baseclasses.processing_parameters import ProcessingParameters
from src.configuration.config import get_parser_ibp
from src.monitoring import execution_time_runtime
from src.processing import data_preprocessing, data_postprocessing

logger = logging.getLogger(__name__)


@execution_time_runtime
def distribution(data_name: str) -> Information:
    """
    Processes a CSV file of a specified dataset and distributes the processed data via dataclasses.

    Parameters
    ----------
    data_name : str
        The name of the dataset to process. The CSV file for this dataset should be 
        located in the 'data' directory and follow the naming convention "{data_name}.csv".

    Returns
    -------
    Information
        An instance of the Information dataclass containing the processed data
        and other relevant information. 

    Raises
    ------
    FileNotFoundError
        If the file `{data_name}.csv` does not exist in the 'data' directory.
    """
    parser = get_parser_ibp()
    args = parser.parse_args() 

    cycle_version = 0 # single signal based cycle cutting

    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_path = pathlib.Path(f"output/{data_name}")
    save_path = output_path / id
    save_path.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    with open(f"{save_path}/'parsed_args.json'", 'w') as file:
            ujson.dump(args_dict, file, indent=4)
    
    data_path = pathlib.Path("data")
    data_path = data_path / f"{data_name}.csv"
    separator: str = args.separator
    data: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_path, sep=separator)
    data = clean_header(data=data) 
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0)
    data = data.rename(columns={"Timestamp": "timestamp"})

    if data_name == 'LAB_PLC_BINARY':
        cycle_version = 1 # dual signal based cycle cutting
    data = data.replace({False: 0, True: 1})
    data.reset_index(inplace=True, drop=True)
    
    parameters = ProcessingParameters(
        dynamic_variance=args.dynamic_variance,
        dynamic_threshold=args.dynamic_threshold,
        correlation_threshold=args.correlation_threshold,
        gamma_threshold=args.gamma_threshold,   
        data_name=args.data_set,
        sampling_frequency_unit=args.sampling_frequency_unit,
        sampling_frequency_round=args.sampling_frequency_round,
    )
    uniform_data, event_based_data, frequency = data_preprocessing(data=data, params=parameters) 
    uniform_data = data_postprocessing(data=uniform_data)  

    info = Information(
        uniform_data=uniform_data,
        event_based_data=event_based_data,
        sample_frequency=frequency,
        sampling_frequency_unit=args.sampling_frequency_unit,
        frequency_spectrum_lower_bound=args.frequency_spectrum_lower_bound,
        frequency_spectrum_upper_bound=args.frequency_spectrum_upper_bound,
        pyfftw_threshold=args.pyfftw_threshold,
        peak_rate=args.peak_rate,
        peak_prominence=args.peak_prominence,
        save_path=save_path,
        columns=uniform_data.columns,
        cycle_version=cycle_version,
        data_path=save_path
    )  
    return info


def clean_header(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove umlauts from strings and replaces with the letter+e convention
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame with column names needing to be cleaned
    
    Returns
    -------
    pd.DataFrame : the input DataFrame with cleaned column names
    """
    translation_table = str.maketrans({
        'ü': 'ue', 'Ü': 'Ue',
        'ä': 'ae', 'Ä': 'Ae',
        'ö': 'oe', 'Ö': 'Oe',
        'ß': 'ss', ' ': '_',
        '\"': ''
    })
    data.columns = [col.translate(translation_table) for col in data.columns]
    return data