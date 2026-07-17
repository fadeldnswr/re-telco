"""
Phase 4C: convolutionally coded QPSK-OFDM over
static multipath and AWGN.

Receiver:
    Perfect-CSI ZF equalization
    Hard-input Viterbi path
    Soft-input Viterbi path
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.channel.awgn import AWGNChannel
from src.core.channel.profiles import create_three_path_channel
from src.core.channel.static_multipath import StaticMultipathChannel
from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.coding.interleaver import BlockInterleaver
from src.core.coding.soft_viterbi import SoftViterbiDecoder
from src.core.coding.viterbi import HardViterbiDecoder
from src.core.modulation.qpsk import QPSK
from src.core.ofdm.config import OFDMConfig
from src.core.ofdm.demodulator import OFDMDemodulator
from src.core.ofdm.modulator import OFDMModulator
from src.core.ofdm.resource_grid import ResourceGridMapper
from src.core.receiver.equalizer import ZeroForcingEqualizer
from src.core.system.phase4.simulation_models import Phase4CResult, BitArray, ComplexArray, FloatArray

# Define phase 4C simulation result dataclass
class Phase4CSimulator:
  def __init__(
    self, ofdm_config: OFDMConfig | None = None,
    encoder: ConvolutionalEncoder | None = None,
    hard_decoder: HardViterbiDecoder | None = None,
    soft_decoder: SoftViterbiDecoder | None = None,
    modulation: QPSK | None = None,
    multipath_channel: StaticMultipathChannel | None = None,
    interleaver_seed: int = 2026
  ) -> None:
    self._config = ofdm_config or OFDMConfig()
    self._encoder = encoder or ConvolutionalEncoder()
    self._hard_decoder = hard_decoder or HardViterbiDecoder()
    self._soft_decoder = soft_decoder or SoftViterbiDecoder()
    self._modulation = modulation or QPSK()
    self._multipath_channel = multipath_channel or create_three_path_channel()
    self._grid_mapper = ResourceGridMapper(self._config)
    self._ofdm_modulator = OFDMModulator(self._config)
    self._ofdm_demodulator = OFDMDemodulator(self._config)
    self._equalizer = ZeroForcingEqualizer()
    self._interleaver = BlockInterleaver.create(length=self._config.qpsk_bits_per_subframe, random_seed=interleaver_seed)
  
  # Define property to access the OFDM configuration
  @property
  def config(self) -> OFDMConfig:
    return self._config
  
  # Define property for information bits per OFDM frame based on the configuration
  @property
  def information_bits_per_frame(self) -> int:
    '''
    For a rate-1/2 encoder with two zero tail bits: coded_length = 2 * (information_length + 2)
    '''
    # Calculate the coded length based on the configuration
    coded_capacity = (
      self._config.qpsk_bits_per_subframe
    )

    # Check if the coded capacity is a multiple of 2 for QPSK modulation
    if coded_capacity % 2 != 0:
      raise ValueError("Coded capacity must be a multiple of 2 for QPSK modulation.")
    
    return (coded_capacity // 2) - self._soft_decoder.encoder_memory  # Subtract 2 for the zero tail bits
  
  # Define method to run the simulation
  def run(self, ebn0_db: float = 10.0, random_seed: int = 42) -> Phase4CResult:
    # Generate random information bits based on the calculated number of bits per frame
    rng = np.random.default_rng(seed=random_seed)

    # Transmitter
    # Generate information bits
    information_bits = rng.integers(
      low=0, high=2, size=self.information_bits_per_frame, dtype=np.uint8
    )

    # Rate 1/2 convolutional encoding of information bits
    encoded_bits = self._encoder.encode(information_bits, terminate=True)

    # Expected coded length based on the encoder's properties
    expected_coded_length = self._config.qpsk_bits_per_subframe

    # Check if the encoded bits length matches the expected coded length
    if encoded_bits.size != expected_coded_length:
      raise ValueError(
        f"Encoded bits length {encoded_bits.size} does not match expected coded length {expected_coded_length}."
      )
    
    # Spread adjacent coded bits across subcarriers and symbols using the resource grid mapper
    interleaved_bits = self._interleaver.interleave(encoded_bits)

    # Transmit the interleaved bits using QPSK modulation
    transmitted_symbols = self._modulation.map(interleaved_bits)

    # Grid the transmitted symbols into the OFDM resource grid
    transmitted_grid = self._grid_mapper.map_symbols(transmitted_symbols)

    # OFDM modulation of the resource grid to generate time-domain samples
    transmitted_waveform = self._ofdm_modulator.modulate(transmitted_grid)

    # Static Multipath Channel
    # Pass the transmitted waveform through the static multipath channel
    multipath_result = self._multipath_channel.transmit(transmitted_waveform, self._config.fft_size)

    # AWGN Channel
    # Create an AWGN channel with the specified Eb/N0
    awgn = AWGNChannel(
      ebn0_db=ebn0_db,
      bits_per_symbol=self._modulation.bits_per_symbol,
      symbol_energy=1.0,
      code_rate=0.5
    )
    result = awgn.transmit_waveform(
      waveform=multipath_result.received_signal,
      number_of_information_bits=information_bits.size,
      rng=rng,
      energy_reference=transmitted_waveform
    )

    # OFDM Receiver
    # OFDM demodulation of the received waveform to obtain the received resource grid
    received_grid = self._ofdm_demodulator.demodulate(result.received_signal)

    # Perfect-CSI ZF equalization of the received resource grid using the channel frequency response
    channel_response_shifted = np.fft.fftshift(
      multipath_result.frequency_response
    )

    # Equalized grid
    equalized_grid = self._equalizer.equalize(
      received_grid, channel_response_shifted
    )

    # Equalized symbols from the resource grid
    equalized_symbols = self._grid_mapper.extract_symbols(equalized_grid)

    # Hard QPSK demapping of equalized symbols to coded bits
    received_interleaved_hard_bits = self._modulation.demap(equalized_symbols)
    received_hard_coded_bits = self._interleaver.deinterleave(received_interleaved_hard_bits)
    hard_decoded_bits = self._hard_decoder.decode(received_hard_coded_bits, terminated=True)

    # Soft QPSK demapping of equalized symbols to LLRs
    active_indices = self._grid_mapper.active_indices()
    active_channel_response = channel_response_shifted[active_indices]

    # ZF output
    # X^[k] = X[k] + W[k] / H[k]
    # Therefore:
    # variance_ZF[k] = N0 / |H[k]|²
    active_zf_noise_variance = result.noise_variance / np.maximum(
      np.abs(
        active_channel_response
      ) ** 2, 1e-12
    )

    # Same channel response for all subframes, so we can use the first subframe's channel response
    symbol_noise_variance = np.tile(
      active_zf_noise_variance, self._config.symbols_per_subframe
    ).astype(np.float64)

    # Soft demapping to LLRs
    received_interleaved_llrs = self._modulation.soft_demap_variable_noise(
      symbols=equalized_symbols, noise_variance=symbol_noise_variance
    )

    # Received soft coded LLRs after deinterleaving
    received_soft_coded_llrs = self._interleaver.deinterleave(received_interleaved_llrs).astype(np.float64)

    # Soft decoding of the received soft coded LLRs
    soft_decoded_bits = self._soft_decoder.decode(received_soft_coded_llrs, terminated=True)

    # Define metrics
    # Check if hard and soft decoded bits have the same length as the original information bits
    if hard_decoded_bits.size != information_bits.size:
      raise ValueError("Hard decoded bits size does not match original information bits size.")
    if soft_decoded_bits.size != information_bits.size:
      raise ValueError("Soft decoded bits size does not match original information bits size.")
    
    # Calculate bit errors for pre-decoder, hard decoded bits, and soft decoded bits
    pre_decoder_bit_errors = int(
      np.count_nonzero(
        encoded_bits != received_hard_coded_bits
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

    # Define minimum and maximum cp lengths based on the multipath channel
    minimum_cp_length = min(
      self._config.cyclic_prefix_length(symbol_index) 
      for symbol_index in range(self._config.symbols_per_subframe)
    )
    maximum_channel_delay = self._multipath_channel.maximum_delay_samples

    # Return the simulation result as a Phase4CResult dataclass
    return Phase4CResult(
      # bits
      information_bits=information_bits,
      encoded_bits=encoded_bits,
      interleaved_bits=interleaved_bits,

      # Symbols
      transmitted_symbols=transmitted_symbols,
      equalized_symbols=equalized_symbols,

      # Received bits and LLRs
      received_interleaved_hard_bits=received_interleaved_hard_bits,
      received_interleaved_llrs=received_interleaved_llrs,
      received_hard_coded_bits=received_hard_coded_bits,
      received_soft_coded_llrs=received_soft_coded_llrs,
      hard_decoded_bits=hard_decoded_bits,
      soft_decoded_bits=soft_decoded_bits,

      # Bit errors
      pre_decoder_bit_errors=pre_decoder_bit_errors,
      hard_payload_bit_errors=hard_payload_bit_errors,
      soft_payload_bit_errors=soft_payload_bit_errors,

      # BERs
      pre_decoder_ber=pre_decoder_bit_errors / encoded_bits.size,
      hard_payload_ber=hard_payload_bit_errors / information_bits.size,
      soft_payload_ber=soft_payload_bit_errors / information_bits.size,

      # Channel parameters
      noise_variance=result.noise_variance,
      maximum_channel_delay=maximum_channel_delay,
      minimum_cp_length=minimum_cp_length,
      cp_condition_satisfied=maximum_channel_delay <= minimum_cp_length
    )