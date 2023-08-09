from typing import List, Set, Tuple
from dataclasses import dataclass, field


@dataclass()
class DetectorOutput:
    """
    Computation of cycle time within a variable range based on cleaned data
    Determination of the strongest cyclic signals
    """
    cycle_time: float = field(repr=False)
    cycle_range: Tuple[float, float] = field(repr=False)
    cyclic_signal_index: int
    cyclic_signal_name: str
    cyclic_signals_indices: Set[int]
    cyclic_signals_names: List[str]

    def __post_init__(self):
        """
        Round cycle_time and values in cycle_range to 4 decimal places after initialization.
        """
        self.cycle_time = round(self.cycle_time, 4)
        self.cycle_range = tuple(round(val, 4) for val in self.cycle_range)

    def __repr__(self):
        return (f'DetectorOutput(cycle_time={self.cycle_time:.4f}, cycle_range=({self.cycle_range[0]:.4f}, '
                f'{self.cycle_range[1]:.4f}), cyclic_signal_index={self.cyclic_signal_index}, '
                f'cyclic_signal_name={self.cyclic_signal_name}, cyclic_signals_indices={self.cyclic_signals_indices}, '
                f'cyclic_signals_names={self.cyclic_signals_names})')
