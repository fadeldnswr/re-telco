'''
Phase 3 simulator module for the digital communications project.
This module provides functionality to simulate the phase 3 of the communication system.
'''
import sys
import numpy as np
from dataclasses import dataclass

from src.core.channel.awgn import AWGNChannel
from src.core.channel.profiles import create_three_path_channel
from src.core.channel.static_multipath import StaticMultipathChannel
from src.core.metrics.communication_metrics import calculate_ber, calculate_evm_rms
from src.core.modulation.qpsk import QPSK
from src.core.ofdm.config import OFDMConfig
from src.core.ofdm.demodulator import OFDMDemodulator
from src.core.ofdm.modulator import OFDMModulator
from src.core.ofdm.resource_grid import ResourceGridMapper
from src.core.receiver.equalizer import MMSEEqualizer, ZeroForcingEqualizer
from src.core.system.phase3.simulation_models import BitArray, ComplexArray, EqualizerType, Phase3Result

from src.exception.exception import CustomException
from src.logging.logging import logger

# Define Phase3Simulator class
class Phase3Simulator:
  def __init__(self, ofdm_config: OFDMConfig | None = None, modulation: QPSK | None = None, multipath_channel: StaticMultipathChannel | None = None) -> None:
    self._config = ofdm_config or OFDMConfig()
    self._modulation = modulation or QPSK()
    self._multipath_channel = multipath_channel or create_three_path_channel()
    self._grid_mapper = ResourceGridMapper(self._config)
    self._ofdm_modulator = OFDMModulator(self._config)
    self._ofdm_demodulator = OFDMDemodulator(self._config)
  
  # Define configuration property
  @property
  def config(self) -> OFDMConfig:
    return self._config
  
  # Define channel property
  @property
  def channel(self) -> StaticMultipathChannel:
    return self._multipath_channel
  
  # Run simulation
  def run(self, ebn0_db: float, equalizer_type: EqualizerType = EqualizerType.ZF, random_seed: int = 42) -> Phase3Result:
    '''
      ebn0_db=None:
        Static multipath without AWGN.
      ebn0_db=float:
        Static multipath followed by AWGN.
    '''
    try:
      # Define random seed for reproducibility
      rng = np.random.default_rng(random_seed)

      # Define transmitted bits
      transmitted_bits = rng.integers(
        low=0, high=2, size=(
          self._config.qpsk_bits_per_subframe
        ), dtype=np.uint8
      )

      # Map bits to symbols
      transmitted_symbols = self._modulation.map(transmitted_bits)

      # Map symbols to resource grid
      transmitted_grid = self._grid_mapper.map_symbols(transmitted_symbols)

      # Modulate the resource grid to generate transmitted waveform
      transmitted_waveform = self._ofdm_modulator.modulate(transmitted_grid)

      # Pass the transmitted waveform through the multipath channel
      multipath_result = (
        self._multipath_channel.transmit(
          signal=transmitted_waveform, fft_size=self._config.fft_size
        )
      )

      # Define received waveform after multipath channel
      multipath_waveform = multipath_result.received_signal

      # Check if ebn0_db is provided for AWGN channel
      if ebn0_db is None:
        # If ebn0_db is None, skip AWGN channel and use multipath waveform directly
        received_waveform = multipath_waveform.copy()
        noise_variance = 0.0
      else:
        # If ebn0_db is provided, pass the multipath waveform through the AWGN channel
        awgn = AWGNChannel(
          ebn0_db=ebn0_db,
          bits_per_symbol=self._modulation.bits_per_symbol,
          symbol_energy=1.0
        )

        # Define received waveform after AWGN channel
        awgn_result = awgn.transmit_waveform(
          waveform=multipath_waveform,
          number_of_information_bits=transmitted_bits.size,
          rng=rng,
          energy_reference=transmitted_waveform
        )

        # Define received waveform and noise variance from AWGN result
        received_waveform = awgn_result.received_signal
        noise_variance = awgn_result.noise_variance
      
      # Demodulate the received waveform to obtain the received resource grid
      received_grid = self._ofdm_demodulator.demodulate(received_waveform)

      # Extract received symbols from the received resource grid
      unequalized_symbols = self._grid_mapper.extract_symbols(received_grid)

      # Channel response shifted
      channel_response_shifted = np.fft.fftshift(multipath_result.frequency_response)

      # Check the equalizer type and perform equalization accordingly
      if equalizer_type == EqualizerType.NONE:
        equalized_grid = received_grid # No equalization applied

      elif equalizer_type == EqualizerType.ZF:
        # Apply Zero Forcing Equalizer to the received grid using the channel response shifted
        equalized_grid = ZeroForcingEqualizer().equalize(
          received_grid, channel_response_shifted
        )

      elif equalizer_type == EqualizerType.MMSE:
        # Apply Minimum Mean Square Error Equalizer to the received grid using the channel response shifted and noise variance
        equalized_grid = MMSEEqualizer().equalize(
          received_grid, channel_response_shifted, noise_variance
        )

      else:
        raise ValueError(f"Invalid equalizer type: {equalizer_type}")
      
      # Extract equalized symbols from the equalized resource grid
      equalized_symbols = self._grid_mapper.extract_symbols(equalized_grid)

      # Demap equalized symbols to recover bits
      recovered_bits = self._modulation.demap(equalized_symbols)

      # Calculate number of bit errors by comparing transmitted and recovered bits
      bit_errors = int(
        np.count_nonzero(
          transmitted_bits != recovered_bits
        )
      )

      # Calculte minimum cyclic prefix length required to avoid ISI based on maximum channel delay
      minimum_cp_length = min(
        self._config.cyclic_prefix_length(symbol_index) for symbol_index in range(self._config.symbols_per_subframe)
      )

      # Maximum delay
      maximum_delay = self._multipath_channel.maximum_delay_samples

      # Return phase 3 result
      return Phase3Result(
        # Bits
        transmitted_bits=transmitted_bits,
        recovered_bits=recovered_bits,

        # Symbols
        transmitted_symbols=transmitted_symbols,
        unequalized_symbols=unequalized_symbols,
        equalized_symbols=equalized_symbols,

        # Waveforms
        transmitted_waveform=transmitted_waveform,
        multipath_waveform=multipath_waveform,
        received_waveform=received_waveform,

        # Channel and impulse response
        impulse_response=multipath_result.impulse_response,
        channel_frequency_response=channel_response_shifted,

        # BER
        bit_errors=bit_errors,
        ber=float(bit_errors / transmitted_bits.size),

        # EVM
        unequalized_evm=calculate_evm_rms(transmitted_symbols, unequalized_symbols),
        equalized_evm=calculate_evm_rms(transmitted_symbols, equalized_symbols),

        # Noise variance and cyclic prefix information
        noise_variance=noise_variance,
        maximum_channel_delay=maximum_delay,
        minimum_cp_length=minimum_cp_length,
        cp_condition_satisfied=(maximum_delay <= minimum_cp_length)
      )
    except Exception as e:
      raise CustomException(e, sys)