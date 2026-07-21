
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

# Define type alias for binary array
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]

# Define a dataclass to hold the results of decoding text using a specific decoder
@dataclass(frozen=True, slots=True)
class DecoderTextResult:
  recovered_bits: BitArray
  recovered_text: str

  bit_errors: int
  ber: float

  frame_errors: int
  frame_error_rate: float

  average_latency_ms: float

  exact_bit_match: bool
  exact_text_match: bool

# Define a dataclass to hold the results of comparing two text decoders
@dataclass(frozen=True, slots=True)
class TextComparisonResult:
  original_text: str
  original_bits: BitArray

  ebn0_db: float
  number_of_frames: int

  viterbi: DecoderTextResult
  neural: DecoderTextResult


# Define a dataclass to hold the aggregate results of a decoder over multiple trials
@dataclass(frozen=True, slots=True)
class DecoderAggregateResult:
  total_bit_errors: int
  total_bits: int
  ber: float

  total_frame_errors: int
  total_frames: int
  frame_error_rate: float

  exact_text_successes: int
  total_trials: int
  exact_text_success_rate: float

  mean_latency_ms: float
  median_latency_ms: float
  latency_p95_ms: float

# Define a dataclass to hold the results of comparing two text decoders at a specific SNR point
@dataclass(frozen=True, slots=True)
class TextSNRResult:
  ebn0_db: float
  viterbi: DecoderAggregateResult
  neural: DecoderAggregateResult

# Define a dataclass to hold the results of a Monte Carlo experiment comparing two text decoders
@dataclass(frozen=True, slots=True)
class TextMonteCarloResult:
  original_text: str
  payload_bits_per_trial: int
  frames_per_trial: int
  trials_per_snr: int
  snr_results: tuple[TextSNRResult, ...]