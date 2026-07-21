"""
Transmit UTF-8 text over convolutionally coded QPSK-AWGN and compare:

1. Soft-input Viterbi
2. Neural decoder

Both decoders receive exactly the same LLR sequence.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from src.core.channel.awgn import AWGNChannel
from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.coding.soft_viterbi import SoftViterbiDecoder
from src.core.modulation.qpsk import QPSK
from src.core.source.bit_framing import reconstruct_payload_bits, segment_bits
from src.core.source.text_codec import TextCodec
from src.ml.models.service import NeuralDecoderService
from src.ml.evaluation.evaluation_models import DecoderTextResult, TextComparisonResult, BitArray
from src.exception.exception import CustomException

# Define the main class for comparing text decoders
class TextDecoderComparison:
  information_bits_per_frame: int = 1006
  nominal_code_rate: float = 1 / 2

  # Define constructor to initialize the text decoder comparison with a neural decoder service
  def __init__(self, neural_model_path: str | Path) -> None:
    self._encoder = ConvolutionalEncoder()
    self._qpsk = QPSK()
    self._viterbi = SoftViterbiDecoder()
    self._neural = NeuralDecoderService(model_path=neural_model_path)

    # Warm up tensorflow by running a dummy inference
    dummy_llrs = np.zeros(
      2 * (self.information_bits_per_frame + 2),
      dtype=np.float64
    )

    # Run a dummy inference to warm up the neural decoder
    self._neural.decode(dummy_llrs, terminated=True)
  
  # Define static method to build result
  @staticmethod
  def _build_result(original_text: str, original_bits: BitArray, recovered_bits: BitArray, frame_errors: int, number_of_frames: int, total_latency_seconds: float) -> DecoderTextResult:
    # Define bit errors and bit error rate
    bit_errors = int(np.count_nonzero(
      original_bits != recovered_bits
    ))

    # Define recovered text by decoding the recovered bits using UTF-8
    recovered_text = TextCodec.decode(recovered_bits, errors="replace")

    # Return the result of decoding text using a specific decoder
    return DecoderTextResult(
      recovered_bits=recovered_bits,
      recovered_text=recovered_text,
      bit_errors=bit_errors,
      ber=bit_errors / original_bits.size,
      frame_errors=frame_errors,
      frame_error_rate=frame_errors / number_of_frames,
      average_latency_ms=1000 * total_latency_seconds / number_of_frames,
      exact_bit_match=bool(np.array_equal(
        original_bits, recovered_bits
      )),
      exact_text_match=bool(original_text == recovered_text)
    )
  
  # Define transmit method to transmit text over convolutionally coded QPSK-AWGN and compare decoders
  def transmit(self, text: str, ebn0_db: float, random_seed: int = 42) -> TextComparisonResult:
    # Define original bits
    original_bits = TextCodec.encode(text)

    # Define frames
    frames = segment_bits(payload_bits=original_bits, frame_capacity=self.information_bits_per_frame)

    # Define list to hold the results of decoding text using Viterbi and neural decoders
    viterbi_recovered_frames: list[BitArray] = []
    neural_recovered_frames: list[BitArray] = []

    # Define variable to hold the total latency for Viterbi and neural decoders
    viterbi_frame_errors = 0
    neural_frame_errors = 0
    total_viterbi_time = 0.0
    total_neural_time = 0.0

    # Define random number generator for reproducibility
    rng = np.random.default_rng(seed=random_seed)

    # Iterate through each frame and perform encoding, modulation, transmission, demodulation, and decoding
    for frame in frames:
      # Define common transmitter and channel path
      encoded_bits = self._encoder.encode(frame.bits, terminate=True)
      expected_encoded_length = 2 * (self.information_bits_per_frame + 2)

      # Check if the length of encoded bits is equal to the expected encoded length
      if encoded_bits.size != expected_encoded_length:
        raise RuntimeError(
          f"Encoded bits length {encoded_bits.size} does not match expected length {expected_encoded_length}."
        )
      
      # Modulate the encoded bits using QPSK modulation
      transmitted_symbols = self._qpsk.map(encoded_bits)

      # Transmit the modulated symbols over an AWGN channel
      channel = AWGNChannel(
        ebn0_db=ebn0_db,
        bits_per_symbol=self._qpsk.bits_per_symbol,
        symbol_energy=1.0,
        code_rate=self.nominal_code_rate,
      )

      # Transmit the symbols through the channel and obtain the received symbols
      received_symbols = channel.transmit(transmitted_symbols, rng=rng)

      # Demodulate the received symbols to obtain log-likelihood ratios (LLRs) for the bits
      llrs = self._qpsk.soft_demap(received_symbols, noise_variance=channel.noise_variance)

      # Soft viterbi decoding
      viterbi_start = perf_counter()
      viterbi_bits = self._viterbi.decode(llrs, terminated=True)
      total_viterbi_time += perf_counter() - viterbi_start

      # Neural decoding
      neural_start = perf_counter()
      neural_bits = self._neural.decode(llrs, terminated=True)
      total_neural_time += perf_counter() - neural_start

      # Compare only the payload bits (excluding tail bits) for frame error counting
      original_valid = frame.bits[:frame.valid_bit_count]
      viterbi_valid = viterbi_bits[:frame.valid_bit_count]
      neural_valid = neural_bits[:frame.valid_bit_count]

      # Check if the array are equal
      if not np.array_equal(original_valid, viterbi_valid):
        viterbi_frame_errors += 1
      if not np.array_equal(original_valid, neural_valid):
        neural_frame_errors += 1
      
      # Append the recovered bits for Viterbi and neural decoders to their respective lists
      viterbi_recovered_frames.append(viterbi_bits)
      neural_recovered_frames.append(neural_bits)
    
    # Remove frame padding and reconstruct the original payload length for Viterbi and neural decoders
    viterbi_recovered_bits = reconstruct_payload_bits(viterbi_recovered_frames, frames)
    neural_recovered_bits = reconstruct_payload_bits(neural_recovered_frames, frames)

    # Return result
    return TextComparisonResult(
      original_text=text,
      original_bits=original_bits,
      ebn0_db=ebn0_db,
      number_of_frames=len(frames),
      viterbi=self._build_result(
        original_text=text,
        original_bits=original_bits,
        recovered_bits=viterbi_recovered_bits,
        frame_errors=viterbi_frame_errors,
        number_of_frames=len(frames),
        total_latency_seconds=total_viterbi_time
      ),
      neural=self._build_result(
        original_text=text,
        original_bits=original_bits,
        recovered_bits=neural_recovered_bits,
        frame_errors=neural_frame_errors,
        number_of_frames=len(frames),
        total_latency_seconds=total_neural_time
      )
    )