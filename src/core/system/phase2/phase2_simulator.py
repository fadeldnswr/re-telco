'''
Phase 2 simulator module.
This module provides functionality to simulate the phase 2 of the communication system.
'''
import numpy as np
import sys

from dataclasses import dataclass
from numpy.typing import NDArray

from src.core.ofdm.config import OFDMConfig
from src.core.ofdm.modulator import OFDMModulator
from src.core.ofdm.demodulator import OFDMDemodulator
from src.core.ofdm.resource_grid import ResourceGridMapper
from src.core.modulation.qpsk import QPSK
from src.core.system.phase2.simulation_models import (
  Phase2SimulatorResult, Phase2BResult, Phase2BBERResult
)
from src.core.channel.awgn import AWGNChannel
from src.core.metrics.communication_metrics import (
  calculate_ber, calculate_evm_rms
)

from src.exception.exception import CustomException
from src.logging.logging import logger

# Define Phase2Simulator class
class Phase2Simulator:
  def __init__(self, ofdm_config: OFDMConfig | None = None, modulation: QPSK | None = None) -> None:
    self._config = ofdm_config or OFDMConfig()
    self._modulation = modulation or QPSK()
    self._grid_mapper = ResourceGridMapper(self._config)
    self._ofdm_modulator = OFDMModulator(self._config)
    self._ofdm_demodulator = OFDMDemodulator(self._config)
  
  # Define class config property
  @property
  def config(self) -> OFDMConfig:
    return self._config
  
  # Run simulation
  def run(self, random_seed: int = 42, ebn0_db: float = 6.0) -> Phase2BResult:
    '''Run the phase 2 simulation and return the results'''
    try:
      # Define random seed for reproducibility
      rnd = np.random.default_rng(random_seed)

      # Define transmitted bits
      transmitted_bits = rnd.integers(
        low=0, high=2, 
        size=self._config.qpsk_bits_per_subframe, dtype=np.uint8
      )

      # Define transmitted symbols by mapping bits
      transmitted_symbols = self._modulation.map(transmitted_bits)

      # Define transmitted symbols by mapping bits
      transmitted_grid = self._grid_mapper.map_symbols(
        transmitted_symbols
      )

      # Define transmitted waveform by modulating the resource grid
      transmitted_waveform = self._ofdm_modulator.modulate(transmitted_grid)

      # Add AWGN noise to the transmitted waveform
      channel = AWGNChannel(
        ebn0_db=ebn0_db,
        bits_per_symbol=self._modulation.bits_per_symbol,
        symbol_energy=1.0
      )

      # Channel result
      channel_result = channel.transmit_waveform(
        waveform=transmitted_waveform,
        number_of_information_bits=transmitted_bits.size,
        rng=rnd
      )

      # Define received waveform (assuming ideal channel, no noise)
      received_waveform = channel_result.received_signal

      # Define received grid by demodulating the received waveform
      recovered_grid = self._ofdm_demodulator.demodulate(received_waveform)

      # Define recovered symbols by extracting active subcarriers from the grid
      recovered_symbols = self._grid_mapper.extract_symbols(recovered_grid)

      # Define recovered bits by demapping symbols
      recovered_bits = self._modulation.demap(recovered_symbols)

      # Define number of bit errors by comparing transmitted and recovered bits
      number_of_bit_errors = int(np.count_nonzero(
        transmitted_bits != recovered_bits
      ))

      # Return the simulation results as a Phase2SimulatorResult dataclass
      return Phase2BResult(
        transmitted_bits=transmitted_bits,
        recovered_bits=recovered_bits,
        transmitted_symbols=transmitted_symbols,
        recovered_symbols=recovered_symbols,
        transmitted_grid=transmitted_grid,
        recovered_grid=recovered_grid,
        transmitted_waveform=transmitted_waveform,
        received_waveform=received_waveform,
        noise=channel_result.noise,
        number_of_bit_errors=number_of_bit_errors,
        ber=calculate_ber(transmitted_bits, recovered_bits),
        evm_rms=calculate_evm_rms(transmitted_symbols, recovered_symbols),
        waveform_bit_energy=channel_result.bit_energy,
        noise_variance=channel_result.noise_variance,
        noise_standard_deviation=channel_result.noise_std_dev,
        waveform_length=int(transmitted_waveform.size)
      )
    except Exception as e:
      raise CustomException(e, sys)
  
  # Define method to run BER simulation over a range of Eb/N0 values
  def run_ber_experiment(self, ebn0_db: float, number_of_subframes: int = 500, random_seed: int = 42) -> Phase2BBERResult:
    '''Run a BER experiment over a range of Eb/N0 values'''
    try:
      # Check if the number of subframes is positive
      if number_of_subframes <= 0:
        raise ValueError("Number of subframes must be a positive integer.")
      
      # Define total bits and total bit errors
      total_bits = 0
      total_bit_errors = 0

      # Iterate over the number of subframes to simulate
      for i in range(number_of_subframes):
        # Run the simulation for each subframe
        result = self.run(random_seed=random_seed + i, ebn0_db=ebn0_db)
        total_bits += result.transmitted_bits.size
        total_bit_errors += result.number_of_bit_errors
      
      # Return the BER experiment results as a Phase2BBERResult dataclass
      return Phase2BBERResult(
        ebn0_db=ebn0_db,
        total_subframes=number_of_subframes,
        total_bits=total_bits,
        total_bit_errors=total_bit_errors,
        ber=float(total_bit_errors / total_bits)
      )
    except Exception as e:
      raise CustomException(e, sys)