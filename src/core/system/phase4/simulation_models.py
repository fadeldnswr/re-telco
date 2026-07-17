from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Define type aliases for arrays
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# Define a dataclass to hold the results of the Phase 4B simulation
@dataclass(frozen=True, slots=True)
class Phase4BResult:
  information_bits: BitArray
  encoded_bits: BitArray

  transmitted_symbols: ComplexArray
  received_symbols: ComplexArray

  hard_received_coded_bits: BitArray
  soft_received_llrs: FloatArray

  hard_decoded_bits: BitArray
  soft_decoded_bits: BitArray

  coded_channel_bit_errors: int
  hard_payload_bit_errors: int
  soft_payload_bit_errors: int

  coded_channel_ber: float
  hard_payload_ber: float
  soft_payload_ber: float

  noise_variance: float
  effective_code_rate: float

# Define a dataclass to hold the results of the Phase 4C simulation
@dataclass(frozen=True, slots=True)
class Phase4CResult:
  information_bits: BitArray
  encoded_bits: BitArray
  interleaved_bits: BitArray

  transmitted_symbols: ComplexArray
  equalized_symbols: ComplexArray

  received_interleaved_hard_bits: BitArray
  received_interleaved_llrs: FloatArray

  received_hard_coded_bits: BitArray
  received_soft_coded_llrs: FloatArray

  hard_decoded_bits: BitArray
  soft_decoded_bits: BitArray

  pre_decoder_bit_errors: int
  hard_payload_bit_errors: int
  soft_payload_bit_errors: int

  pre_decoder_ber: float
  hard_payload_ber: float
  soft_payload_ber: float

  noise_variance: float
  maximum_channel_delay: int
  minimum_cp_length: int
  cp_condition_satisfied: bool