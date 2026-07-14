'''
Simulation of the first phase of the system.
'''
import numpy as np
import os
import sys

from dataclasses import dataclass
from numpy.typing import NDArray

from src.core.channel.awgn import AWGNChannel
from src.core.metrics.communication_metrics import (
  average_symbol_energy,
  calculate_ber, calculate_evm_rms
)
from src.core.modulation.qpsk import QPSK
from src.core.system.config import SimulationConfig
from src.exception.exception import CustomException
from src.logging.logging import logger

# Define bit array and complex array types for better type hinting
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]

# Define a dataclass to hold the results of the phase 1 simulation
@dataclass(frozen=True, slots=True)
class Phase1SimulationResult:
  transmitted_bits: BitArray
  recovered_bits: BitArray

  transmitted_symbols: ComplexArray
  received_symbols: ComplexArray

  number_of_bit_errors: int
  ber: float
  evm_rms: float

  average_tx_symbol_energy: float
  noise_variance: float

# Define the Phase1Simulator class to encapsulate the simulation logic
class Phase1Simulator:
  def __init__(self, modulation: QPSK | None = None) -> None:
    self._modulation = modulation or QPSK()
  
  # Define method to run the phase 1 simulation
  def run(self, config: SimulationConfig) -> Phase1SimulationResult:
    try:
      # Generate random bits for transmission
      rng = np.random.default_rng(config.random_seed)

      # Generate random bits for transmission
      transmitted_bits = rng.integers(
        low=0, high=2, size=config.number_of_bits, dtype=np.uint8
      )

      # Modulate the bits into symbols
      transmitted_symbols = self._modulation.map(transmitted_bits)

      # Define channel parameters based on the configuration
      channel = AWGNChannel(
        ebn0_db=config.ebn0_db,
        bits_per_symbol=self._modulation.bits_per_symbol,
        symbol_energy=1.0
      )

      # Define received symbols
      received_symbols = channel.transmit(transmitted_symbols, rng)

      # Demodulate the received symbols back into bits
      recovered_bits = self._modulation.demap(received_symbols)

      # Calculate the number of bit errors and Bit Error Rate (BER)
      number_of_bit_errors = np.count_nonzero(
        transmitted_bits != recovered_bits
      )

      # Return the results of the simulation encapsulated in a Phase1SimulationResult dataclass
      return Phase1SimulationResult(
        transmitted_bits=transmitted_bits,
        recovered_bits=recovered_bits,
        transmitted_symbols=transmitted_symbols,
        received_symbols=received_symbols,
        number_of_bit_errors=number_of_bit_errors,
        ber=calculate_ber(
          transmitted_bits,
          recovered_bits,
        ),
        evm_rms=calculate_evm_rms(
          transmitted_symbols,
          received_symbols,
        ),
        average_tx_symbol_energy=average_symbol_energy(
          transmitted_symbols
        ),
        noise_variance=channel.noise_variance,
      )
    except Exception as e:
      raise CustomException(e, sys)