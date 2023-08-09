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
    
    detection = Detector(info=info)
    output = detection.run()
    logger.info(f"{output}")
    
    cycled_data, cycle_times = cut_cycles(data=info.event_based_data,
                                cyclic_signal=output.cyclic_signal_name,
                                true_cycle_time=output.cycle_time,
                                edge_condition=args.cyclic_signal_edge)
    
    logger.info(f"amount of cycles in the cut data: {cycled_data.iloc[:, -1].max()}")


if __name__ == "__main__":
    main()