'''
Simulation models for Phase 2 of the digital communications system.
'''

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

# Define complex array type for better type hinting
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]

# Define Phase2Simulator class result
@dataclass(frozen=True, slots=True)
class Phase2SimulatorResult:
  transmitted_bits: BitArray
  recovered_bits: BitArray

  transmitted_symbols: ComplexArray
  recovered_symbols: ComplexArray
  transmitted_waveform: ComplexArray

  transmitted_grid: ComplexArray
  recovered_grid: ComplexArray

  number_of_bit_errors: int
  ber: float

  maximum_grid_error: float
  mean_grid_error: float
  waveform_length: int

# Define Phase2BResult dataclass for Phase 2B simulation results
@dataclass(frozen=True, slots=True)
class Phase2BResult:
  transmitted_bits: BitArray
  recovered_bits: BitArray

  transmitted_symbols: ComplexArray
  recovered_symbols: ComplexArray

  transmitted_grid: ComplexArray
  recovered_grid: ComplexArray

  transmitted_waveform: ComplexArray
  received_waveform: ComplexArray
  noise: ComplexArray

  number_of_bit_errors: int
  ber: float
  evm_rms: float

  waveform_bit_energy: float
  noise_variance: float
  noise_standard_deviation: float
  waveform_length: int

# Define Phase2BBerResult dataclass for Phase 2B BER simulation results
@dataclass(frozen=True, slots=True)
class Phase2BBERResult:
  ebn0_db: float
  total_subframes: int
  total_bits: int
  total_bit_errors: int
  ber: float