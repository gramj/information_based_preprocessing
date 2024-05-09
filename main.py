import logging

from src.configuration.config import get_parser_ibp
from src.data_utils import distribution
from src.cycle_detector import Detector
from src.cycle_cutting import cut_cycles


def main():
    """
    """ 
    parser = get_parser_ibp()
    args = parser.parse_args() 
    
    logger = logging.getLogger("ibp")
    logging.basicConfig(format="%(asctime)s -%(name)s-%(levelname)s- %(message)s", level="INFO")

    data_name = args.data_set
    info = distribution(data_name=data_name)
    
    detection = Detector(info=info, version=info.cycle_version)
    output = detection.run()
    logger.info(f"{output}")
    
    # Cycle Analysis
    cycled_data, cycle_times = cut_cycles(data=info.event_based_data,
                                primary_signal=output.primary_cyclic_signal_name,
                                secondary_signal=output.secondary_cyclic_signal_name,
                                true_cycle_time=output.cycle_time,
                                edge_condition=args.cyclic_signal_edge,
                                version=info.cycle_version
                                )
    logger.info(f"amount of cycles in the cut data: {cycled_data.iloc[:, -1].max()}")

    labels = {
        "LAB_PLC_BINARY": "cycle_time=140.6s, cycles=400, states=14",
    }
    logger.info(f"{data_name} label: {labels.get(data_name, f'{data_name} has no label')}")


if __name__ == "__main__":
    main()