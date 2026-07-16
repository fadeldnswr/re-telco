'''
Define simulation models
'''
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from enum import StrEnum

# Define type aliases for better type hinting
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]

# Define an enumeration for equalizer types
class EqualizerType(StrEnum):
  NONE = "none"
  ZF = "zf"
  MMSE = "mmse"

# Define dataclass for Phase3 simulation result
@dataclass(frozen=True, slots=True)
class Phase3Result:
  transmitted_bits: BitArray
  recovered_bits: BitArray

  transmitted_symbols: ComplexArray
  unequalized_symbols: ComplexArray
  equalized_symbols: ComplexArray

  transmitted_waveform: ComplexArray
  multipath_waveform: ComplexArray
  received_waveform: ComplexArray

  impulse_response: ComplexArray
  channel_frequency_response: ComplexArray

  bit_errors: int
  ber: float

  unequalized_evm: float
  equalized_evm: float

  noise_variance: float
  maximum_channel_delay: int
  minimum_cp_length: int
  cp_condition_satisfied: bool

# Define dataclass for Phase3 BER point
@dataclass(frozen=True, slots=True)
class Phase3BERPoint:
  ebn0_db: float
  total_bits: int

  no_equalizer_errors: int
  zf_errors: int
  mmse_errors: int

  no_equalizer_ber: float
  zf_ber: float
  mmse_ber: float