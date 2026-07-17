"""
Phase 4B: convolutionally coded QPSK over AWGN.

Compares:
  1. Hard QPSK demapping + hard Viterbi
  2. Soft QPSK demapping + soft Viterbi
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.channel.awgn import AWGNChannel
from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.coding.viterbi import HardViterbiDecoder
from src.core.coding.soft_viterbi import SoftViterbiDecoder
from src.core.modulation.qpsk import QPSK
from src.core.system.phase4.simulation_models import Phase4BResult, BitArray, ComplexArray, FloatArray

# Define phase 4B simulation result dataclass
class Phase4BSimulator:
  def __init__(self, encoder: ConvolutionalEncoder | None = None, modulation: QPSK | None = None, hard_decoder: HardViterbiDecoder | None = None, soft_decoder: SoftViterbiDecoder | None = None):
    self._encoder = encoder or ConvolutionalEncoder()
    self._modulation = modulation or QPSK()
    self._hard_decoder = hard_decoder or HardViterbiDecoder()
    self._soft_decoder = soft_decoder or SoftViterbiDecoder()
  
  # Run the simulation
  def run(self, number_of_information_bits: int = 1000, ebn0_db: float = 3.0, random_seed: int = 42) -> Phase4BResult:
    # Check if the number of information bits is a positive integer
    if number_of_information_bits <= 0:
      raise ValueError("Number of information bits must be a positive integer.")
    
    # Define the random number generator with the given seed
    rng = np.random.default_rng(seed=random_seed)

    # Generate uncoded information bits
    information_bits = rng.integers(
      low=0, high=2, size=number_of_information_bits, dtype=np.uint8
    )

    # Rate 1/2 convolutional encoding of information bits
    encoded_bits = self._encoder.encode(information_bits, terminate=True)

    # Check if encoded bits are a multiple of 2 for QPSK modulation
    if encoded_bits.size % 2 != 0:
      raise ValueError("Encoded bits size must be a multiple of 2 for QPSK modulation.")
    
    # Effective code rate calculation
    effective_code_rate = information_bits.size / encoded_bits.size

    # Map encoded bits to QPSK symbols
    transmitted_symbols = self._modulation.map(encoded_bits)

    # Create an AWGN channel with the specified Eb/N0
    channel = AWGNChannel(
      ebn0_db=ebn0_db,
      bits_per_symbol=self._modulation.bits_per_symbol,
      symbol_energy=1.0,
      code_rate=0.5
    )

    # Add complex AWGN noise to the transmitted symbols
    received_symbols = channel.transmit(transmitted_symbols, rng)

    # Hard QPSK demapping of received symbols to coded bits
    hard_received_coded_bits = self._modulation.demap(received_symbols)
    hard_decoded_bits = self._hard_decoder.decode(hard_received_coded_bits, terminated=True)

    # Soft QPSK demapping of received symbols to LLRs
    soft_received_llrs = self._modulation.soft_demap(received_symbols, noise_variance=channel.noise_variance)
    soft_decoded_bits = self._soft_decoder.decode(soft_received_llrs, terminated=True)

    # Check if both hard and soft decoded bits have the same length as the original information bits
    if hard_decoded_bits.size != information_bits.size:
      raise ValueError("Hard decoded bits size does not match original information bits size.")
    if soft_decoded_bits.size != information_bits.size:
      raise ValueError("Soft decoded bits size does not match original information bits size.")
    
    # Calculate bit errors for coded channel, hard decoded bits, and soft decoded bits
    coded_channel_bit_errors = int(
      np.count_nonzero(
        encoded_bits != hard_received_coded_bits
      )
    )
    hard_payload_bit_errors = int(
      np.count_nonzero(
        information_bits != hard_decoded_bits
      )
    )
    soft_payload_bit_errors = int(
      np.count_nonzero(
        information_bits != soft_decoded_bits
      )
    )

    # Return result
    return Phase4BResult(
      # bits
      information_bits=information_bits,
      encoded_bits=encoded_bits,

      # Symbols
      transmitted_symbols=transmitted_symbols,
      received_symbols=received_symbols,

      # Received bits and LLRs
      hard_received_coded_bits=hard_received_coded_bits,
      soft_received_llrs=soft_received_llrs,
      hard_decoded_bits=hard_decoded_bits,
      soft_decoded_bits=soft_decoded_bits,
      coded_channel_bit_errors=coded_channel_bit_errors,
      hard_payload_bit_errors=hard_payload_bit_errors,
      soft_payload_bit_errors=soft_payload_bit_errors,

      # Bit error rates
      coded_channel_ber=coded_channel_bit_errors / encoded_bits.size,
      hard_payload_ber=hard_payload_bit_errors / information_bits.size,
      soft_payload_ber=soft_payload_bit_errors / information_bits.size,
      noise_variance=channel.noise_variance,
      effective_code_rate=effective_code_rate
    )