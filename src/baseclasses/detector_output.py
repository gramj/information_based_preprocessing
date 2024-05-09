from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass()
class DetectorOutput:
    """
    Computation of cycle time within a variable range based on cleaned data
    Determination of the strongest cyclic signals
    """
    cycle_time: float = field(repr=False)
    cycle_range: Tuple[float, float] = field(repr=False)
    primary_cyclic_signal_index: int
    primary_cyclic_signal_name: str
    secondary_cyclic_signal_index: int
    secondary_cyclic_signal_name: str
    cyclic_signals_indices: Optional[Set[int]] = None
    cyclic_signals_names: Optional[List[str]] = None

    def __post_init__(self):
        """
        Round cycle_time and values in cycle_range to 4 decimal places after initialization.
        """
        self.cycle_time = round(self.cycle_time, 4)
        self.cycle_range = tuple(round(val, 3) for val in self.cycle_range)

    def __repr__(self):
        return (f'DetectorOutput(cycle_time={self.cycle_time:.4f}, cycle_range=({self.cycle_range[0]:.4f}, '
                f'{self.cycle_range[1]:.4f}), primary_cyclic_signal_index={self.primary_cyclic_signal_index}, '
                f'primary_cyclic_signal_name="{self.primary_cyclic_signal_name}", '
                f'secondary_cyclic_signal_index={self.secondary_cyclic_signal_index}, '
                f'secondary_cyclic_signal_name="{self.secondary_cyclic_signal_name}", '
                f'cyclic_signals_indices={self.cyclic_signals_indices}, '
                f'cyclic_signals_names={self.cyclic_signals_names})')
